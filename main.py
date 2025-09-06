# app.py

import os
import streamlit as st
from io import BytesIO
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from dotenv import load_dotenv

# Load environment variables from API.env file
load_dotenv('API.env')


def summarize_pdf(pdf_file_path, custom_prompt_text):
    """
    Summarizes a PDF using a user-provided prompt with Gemini.
    """
    # 1. Instantiate LLM model
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found. Please set it in your API.env file.")
        return None

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash-latest",
        temperature=0.3,
        google_api_key=api_key
    )

    # 2. Load and split the PDF
    loader = PyPDFLoader(pdf_file_path)
    docs_chunks = loader.load_and_split(
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=20000, chunk_overlap=1000)
    )

    # 3. Create the prompt from the user's template
    prompt_template = custom_prompt_text + """

    {text}

    """
    prompt = PromptTemplate.from_template(prompt_template)

    # 4. Create and run the summarization chain
    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)

    result = chain.invoke({"input_documents": docs_chunks})

    return result['output_text']


def main():
    st.set_page_config(page_title="Custom PDF Summarizer", page_icon="✍️", layout="wide")

    # Header with styling
    st.markdown(
        """
        <h1 style="text-align:center;">✍️ Custom PDF Summarizer with Gemini</h1>
        <p style="text-align:center; font-size:18px; color:gray;">
            Upload a PDF, provide a custom instruction, and get a tailored summary powered by Gemini.
        </p>
        """,
        unsafe_allow_html=True,
    )

    # Upload section
    uploaded_file = st.file_uploader("📂 **Step 1: Upload your PDF file**", type="pdf")

    if uploaded_file is not None:
        temp_file_path = os.path.join(".", "temp_uploaded_file.pdf")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.success(f"✅ Successfully uploaded: `{uploaded_file.name}`")

        # Prompt input
        custom_prompt = st.text_area(
            "📝 **Step 2: Enter your custom prompt**",
            height=150,
            placeholder="Example: Summarize this research paper in simple terms with 5 bullet points."
        )

        # Generate summary button
        if st.button("🚀 Generate Summary", type="primary", use_container_width=True):
            if not custom_prompt:
                st.warning("⚠️ Please enter a prompt to guide the summary.")
            else:
                with st.spinner("🧠 Gemini is generating your summary..."):
                    try:
                        summary = summarize_pdf(temp_file_path, custom_prompt)
                        if summary:
                            st.subheader("📌 Your Custom Summary")
                            st.write(summary)

                            # Download button for summary
                            b = BytesIO(summary.encode("utf-8"))
                            st.download_button(
                                label="📥 Download Summary",
                                data=b,
                                file_name="summary.txt",
                                mime="text/plain"
                            )

                    except Exception as e:
                        st.error(f"❌ An error occurred: {e}")

        # Clean up temp file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


if __name__ == "__main__":
    main()
