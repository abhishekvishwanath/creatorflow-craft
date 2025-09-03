import os
from dotenv import load_dotenv
import streamlit as st

# Optional: If using LangGraph or Groq API
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq.chat_models import ChatGroq  # or local HuggingFace wrapper

# Load environment variables
load_dotenv()

# ------------------------------
# Streamlit UI
# ------------------------------
st.set_page_config(page_title="Social Media Script Generator", layout="wide")
st.title("Social Media Script Generator")

# Sidebar inputs
st.sidebar.header("Agent Settings")
platform = st.sidebar.selectbox(
    "Select Social Media Platform", 
    ["YouTube", "Instagram", "Twitter", "LinkedIn"]
)
length = st.sidebar.selectbox(
    "Select Script Length", 
    ["Short", "Medium", "Long"]
)

topic = st.text_input("Enter the topic/prompt for the script:")

# ------------------------------
# Prompt template (system + user)
# ------------------------------
prompt = ChatPromptTemplate([
    ("system", "You are a professional social media content writer."),
    ("user", """
Generate a {length} script for a {platform} post/video about the following topic:

Topic: {topic}

Ensure the content is engaging, clear, and suitable for the selected platform.
""")
])

# Output parser
parser = StrOutputParser()

# Response function using chain
def generate_script(topic, platform, length):
    # If using Groq API
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(model="gemma2-9b-it", api_key=groq_api_key)
    
    # Build chain: prompt -> llm -> parser
    chain = prompt | llm | parser
    
    # Invoke chain with dynamic inputs
    response = chain.invoke({
        "topic": topic,
        "platform": platform,
        "length": length
    })
    return response

# ------------------------------
# Streamlit button
# ------------------------------
if st.button("Generate Script"):
    if not topic:
        st.warning("Please enter a topic/prompt.")
    else:
        script = generate_script(topic, platform, length)
        st.subheader("Generated Script")
        st.write(script)
