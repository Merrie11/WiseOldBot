import streamlit as st
import openai
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# 🔥 Zet hier direct je API-key (LET OP: Dit is niet veilig voor productie!)
OPENAI_API_KEY = "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"

# ✅ Check of API-key correct is geladen
if not OPENAI_API_KEY or not OPENAI_API_KEY.startswith("sk-"):
    st.error("⚠ ERROR: Ongeldige API-key. Zorg dat je de juiste key hier invoert!")
    st.stop()

# 🚀 Laad AI-model en database
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    index = faiss.read_index("osrs_index.bin")
    articles = np.load("osrs_articles.npy", allow_pickle=True)
except Exception as e:
    st.error(f"⚠ Error loading FAISS or OSRS data: {str(e)}")
    st.stop()

# 🔍 Functie om OSRS Wiki te doorzoeken
def search_osrs_wiki(query, top_k=3):
    query_embedding = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    results = []
    for idx in indices[0]:
        title, content = articles[idx]
        results.append(f"**{title}**\n{content[:800]}...")
    return "\n\n".join(results) if results else "No relevant OSRS data found."

# 🤖 Functie om GPT-4 te gebruiken
def ask_gpt(query, context=""):
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an OSRS expert with a fun and engaging personality."},
                {"role": "user", "content": f"{context}\n\nUser: {query}"}
            ]
        )
        return response.choices[0].message.content
    except openai.AuthenticationError:
        return "⚠ Error: Invalid API Key. Please check your OpenAI credentials."
    except Exception as e:
        return f"⚠ An unexpected error occurred: {str(e)}"

# 🔥 Streamlit UI - Chat Style
st.set_page_config(page_title="WiseOldBot - OSRS AI", layout="wide")
st.title("🧙‍♂️ WiseOldBot - The OSRS AI Chatbot")
st.write("Ask anything about Old School RuneScape and I'll provide expert advice!")

# 📝 Gespreksgeschiedenis opslaan
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "🛡 **Hey, wanna get your armor trimmed? Just trade me... oh wait, wrong chat. What’s up, scaper?**"}
    ]

# 💬 Toon chatgeschiedenis
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 🏆 User input
user_input = st.chat_input("Type your question here...")

# ℹ️ Disclaimer onder de input
st.markdown(
    "<p style='font-size: 12px; color: gray;'>⚠ WiseOldBot is trained on the OSRS Wiki (CC BY-SA 3.0), but not all articles are included. Some information may be missing or outdated.</p>",
    unsafe_allow_html=True
)

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔍 Zoek eerst in de OSRS Wiki
    wiki_response = search_osrs_wiki(user_input)

    # 📡 Vraag GPT-4 als backup als er geen relevante wiki-data is
    ai_response = wiki_response if wiki_response != "No relevant OSRS data found." else ask_gpt(user_input)

    # 💬 Toon antwoord in de chatgeschiedenis
    st.session_state.chat_history.append({"role": "assistant", "content": ai_response})
    with st.chat_message("assistant"):
        st.markdown(ai_response)
