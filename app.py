from flask import Flask, render_template, request, jsonify, send_from_directory
from sentence_transformers import SentenceTransformer, util
from langchain_groq import ChatGroq
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from gtts import gTTS
import torch
import string
import os
import json
import tempfile
import speech_recognition as sr
from flask import Flask, session


app = Flask(__name__)
app.secret_key = "a_very_secret_key"  # required for sessions

# === Flask App Setup ===

UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'responses')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# === Environment Setup ===
os.environ["GROQ_API_KEY"] = "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"

# === Load Menu ===
with open("menu.json") as f:
    menu = json.load(f)

# === Sentence Transformer ===
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# === LangChain LLM ===
llm = ChatGroq(
    temperature=0.5,
    model_name="llama3-8b-8192"
)


def append_to_history(role, content):
    """Add a message to the session history"""
    if "history" not in session:
        session["history"] = []
    session["history"].append({"role": role, "content": content})
    session.modified = True  # Ensure Flask knows session was modified


# === Menu Embedding ===
def extract_menu_phrases(menu):
    phrases = []
    phrase_to_item = []

    for category, items in menu.items():
        for item in items:
            name = item.get('name', '')
            desc = item.get('description', '')
            price = item.get('price') or item.get('price_from') or 'N/A'
            custom_phrases = [
                f"{name}", f"What is the price of {name}?",
                f"Do you have {name}?", f"Tell me about {name}",
                f"I want to order {name}", f"{name} - {desc} - {price} PKR"
            ]
            for phrase in custom_phrases:
                phrases.append(phrase)
                phrase_to_item.append(name)
    return phrases, phrase_to_item

flat_menu_phrases, phrase_to_item = extract_menu_phrases(menu)
menu_embeddings = embedder.encode(flat_menu_phrases, convert_to_tensor=True)

def get_top_matches(query, top_k=5):
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, menu_embeddings)[0]
    top_indices = torch.topk(similarities, k=top_k).indices

    matched_items = []
    for idx in top_indices:
        item_name = phrase_to_item[idx]
        for cat_items in menu.values():
            for item in cat_items:
                if item["name"] == item_name:
                    matched_items.append(item)
                    break
    return matched_items, similarities[top_indices[0]].item()


def generate_response(user_input):
    """Generate response based on user input and conversation history"""
    # Get conversation history first
    history = session.get("history", [])
    
    # Check if this is a follow-up question (short phrases, references, etc.)
    follow_up_indicators = ["keep it", "the third", "the first", "the second", "number", "yes", "no", "that one", "this one", "add", "remove", "change", "modify"]
    is_follow_up = any(indicator in user_input.lower() for indicator in follow_up_indicators) or len(user_input.split()) < 4
    
    # For follow-up questions, get broader menu context
    if is_follow_up and history:
        # Look for recently mentioned menu categories in conversation
        recent_text = " ".join([msg["content"] for msg in history[-4:]])  # Last 4 messages
        matched_items, similarity_score = get_top_matches(recent_text + " " + user_input)
    else:
        # Normal menu search
        matched_items, similarity_score = get_top_matches(user_input)
    
    # Build system prompt with both matched items and conversation context
    item_info = json.dumps(matched_items, indent=2) if matched_items else "No matched items found."
    
    # Include recent conversation context in system prompt
    recent_context = ""
    if history:
        recent_messages = history[-6:]  # Last 6 messages for context
        recent_context = "\n\nRecent conversation:\n"
        for msg in recent_messages:
            role = "Customer" if msg["role"] == "user" else "Assistant"
            recent_context += f"{role}: {msg['content']}\n"

    messages = [
        SystemMessage(content=f"""
You are a polite and helpful restaurant assistant.

Here are the matched menu items the customer might be referring to:
{item_info}

{recent_context}

IMPORTANT: 
1. When the customer refers to "the third one", "the first option", "number 2", etc., look at the recent conversation context above to understand what list they're referring to.
2. When the customer says follow-up phrases like "keep it as is", "add cheese", "make it spicy", etc., they are referring to items discussed in the recent conversation.
3. Use the conversation context to understand what category/items they're talking about.

Only answer questions based on the menu items and previous conversation context. Do not invent or guess anything. 
Be conversational and helpful. Always maintain context from the previous conversation.
""")
    ]

    # Add chat history to provide context
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "bot":
            messages.append(AIMessage(content=msg["content"]))

    # Add current user input
    messages.append(HumanMessage(content=user_input))

    # Get LLM response
    response = llm(messages)
    bot_reply = response.content.strip()

    # Save both user input and bot response to session history
    append_to_history("user", user_input)
    append_to_history("bot", bot_reply)

    return bot_reply

def transcribe_audio(file_path):
    """Transcribe audio file to text"""
    r = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio = r.record(source)
    return r.recognize_google(audio)

def synthesize_speech(text):
    """Convert text to speech and save as audio file"""
    # Remove punctuation for better TTS
    text = text.translate(str.maketrans('', '', string.punctuation))
    tts = gTTS(text=text, lang='en', slow=False)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3", dir=UPLOAD_FOLDER)
    tts.save(temp_file.name)
    filename = os.path.basename(temp_file.name)
    return filename

# === Routes ===
@app.route("/")
def home():
    """Home page - clears session history"""
    session.pop("history", None)
    return render_template("index.html")

@app.route("/process_audio", methods=["POST"])
def process_audio():
    """Process audio input and return transcribed text and bot response"""
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files['audio']
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
    audio_file.save(temp_path)

    try:
        # Transcribe audio to text
        user_text = transcribe_audio(temp_path)
        print(f"Transcribed text: {user_text}")  # Debug logging
        
        # Generate bot response
        bot_response = generate_response(user_text)
        print(f"Bot response: {bot_response}")  # Debug logging
        
        # Convert response to speech
        audio_filename = synthesize_speech(bot_response)

        # Clean up temporary file
        os.unlink(temp_path)

        return jsonify({
            "user_text": user_text,
            "bot_text": bot_response,
            "audio_url": f"/static/responses/{audio_filename}"
        })
    except Exception as e:
        print(f"Error processing audio: {str(e)}")  # Debug logging
        # Clean up temporary file even on error
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        return jsonify({"error": str(e)}), 500

@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Clear conversation history"""
    session.pop("history", None)
    return jsonify({"message": "History cleared"})

@app.route("/static/responses/<path:filename>")
def serve_audio(filename):
    """Serve audio files"""
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)