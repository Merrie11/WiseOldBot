import os
import streamlit as st
import openai
import faiss
import requests
import numpy as np
from sentence_transformers import SentenceTransformer

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def download_file(url, filename):
    """Download een bestand als het nog niet bestaat."""
    if not os.path.exists(filename):
        print(f"📥 Downloading {filename}...")
        response = requests.get(url, stream=True)
        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"✅ {filename} downloaded!")

# 🔥 Download alleen osrs_index.bin als het niet aanwezig is
BIN_URL = "https://github.com/Merrie11/WiseOldBot/blob/main/Backup/osrs_index.bin"
download_file(BIN_URL, "osrs_index.bin")

index = faiss.read_index("osrs_index.bin")
articles = np.load("osrs_articles.npy", allow_pickle=True)

try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("osrs_index.bin")
    articles = np.load("osrs_articles.npy", allow_pickle=True)
except Exception as e:
    st.error(f"⚠ Error loading FAISS or OSRS data: {str(e)}")
    st.stop()

def search_osrs_wiki(query, top_k=3):
    try:
        query_embedding = model.encode([query], convert_to_numpy=True)
        distances, indices = index.search(query_embedding, top_k)
        results = []
        for idx in indices[0]:
            title, content = articles[idx]
            results.append(f"**{title}**\n{content[:800]}...")
        return "\n\n".join(results) if results else "No relevant OSRS data found."
    except Exception:
        return "⚠ Error retrieving OSRS data."

def generate_funny_response(user_input):
    prompt = f"""
    You are WiseOldBot, an OSRS veteran with a sarcastic but fun personality.
    If the user types something that isn't a real question, generate a funny response.

    **Examples:**
    - If they type gibberish, respond with: "Ah yes, ancient Zarosian... or just keyboard smashing?"
    - If they type one random word, joke about it: "Just 'fish'? Are you trying to become a monkfish IRL?"
    - If they are being weird, assume they fell for a doubling scam.

    **User Input:** "{user_input}"
    **Your Funny Response:**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "system", "content": "You are a witty OSRS veteran with a fun, sarcastic personality."},
                      {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠ Error: {str(e)}"

def ask_gpt(query, context):
    prompt = f"""
    You are WiseOldBot, a witty and knowledgeable OSRS veteran.
    Keep your responses engaging, humorous, and useful.

    **Previous conversation:**
    {context}

    **User's latest question:**
    {query}

    **Your response:**
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an OSRS expert with a fun and engaging personality."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content

    except openai.AuthenticationError:
        return "⚠ Error: Invalid API Key. Please check your OpenAI credentials."

    except Exception as e:
        return f"⚠ An unexpected error occurred: {str(e)}"

st.set_page_config(page_title="WiseOldBot - OSRS AI", layout="wide")

st.title("🧙‍♂️ WiseOldBot - The OSRS AI Chatbot")
st.write("Ask anything about Old School RuneScape and I'll provide expert advice!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "🛡 **Hey, wanna get your armor trimmed? Just trade me... oh wait, wrong chat. What’s up, scaper?**"}
    ]

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Type your question here...")

st.markdown(
    "<p style='font-size: 12px; color: gray; text-align: center;'>⚠ WiseOldBot is trained on the OSRS Wiki (CC BY-SA 3.0), but not all articles are included. Some information may be missing or outdated.</p>",
    unsafe_allow_html=True
)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    context = search_osrs_wiki(user_input)

    if "?" not in user_input and len(user_input.split()) < 5:
        ai_response = generate_funny_response(user_input)
    else:
        ai_response = ask_gpt(user_input, context)

    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
