import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline, AutoModelForSeq2SeqLM, T5Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ----------------------------
# SETUP
# ----------------------------
st.set_page_config(page_title="Smart Research Assistant", page_icon="📝")

with st.sidebar:
    st.title("Smart Research Assistant")

    st.markdown("""
    ### About This Project
    Smart Research Assistant helps you work smarter with research papers.  
    Upload a PDF or text document, generate a concise summary, explore auto-generated questions, and challenge yourself with an interactive quiz.  
    Designed to make understanding and analyzing academic content easier and faster.
    """)

    st.markdown("---")

    st.markdown("### Features")
    st.markdown("""
    • Upload a PDF or TXT document\n
    • Summarize the document\n
    • Generate relevant questions\n
    • Challenge yourself with a quiz
    """)

    st.markdown("---")

    show_preview = st.checkbox("Show Document Preview", key="show_preview")
    
    st.markdown("---")

    st.markdown("""
    Made by Tanishka Singh
    """)


# ----------------------------
# FUNCTIONS
# ----------------------------
def extract_text_from_pdf(uploaded_file):
    text = ""
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    for page in doc:
        text += page.get_text()
    return text

def summarize_text(text):
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    text = text[:3000]
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def generate_questions(context, num_questions=3):
    tokenizer_qg = T5Tokenizer.from_pretrained(
        "mrm8488/t5-base-finetuned-question-generation-ap", use_fast=False
    )
    model_qg = AutoModelForSeq2SeqLM.from_pretrained(
        "mrm8488/t5-base-finetuned-question-generation-ap"
    )

    context = context[:1500].replace("\n", " ")
    input_text = f"generate questions: {context}"
    inputs = tokenizer_qg.encode(input_text, return_tensors="pt", max_length=512, truncation=True)

    outputs = model_qg.generate(
        inputs,
        max_length=64,
        do_sample=True,
        top_k = 50,
        top_p = 0.95,
        temperature = 0.9,
        num_return_sequences=num_questions
    )

    questions = [tokenizer_qg.decode(output, skip_special_tokens=True) for output in outputs]
    questions = [
        q.replace("question:", "").strip()
        for i, q in enumerate(questions)
        if q not in questions[:i] and len(q.split()) > 3 and q.endswith("?")
    ]
    return questions

def correct_grammar(sentence):
    gc_pipeline = pipeline(
        "text2text-generation",
        model="vennify/t5-base-grammar-correction"
    )
    result = gc_pipeline(sentence, max_length=128, do_sample=False)
    return result[0]["generated_text"]

def answer_question(question, context):
    qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")
    result = qa_pipeline(question=question, context=context)
    return result["answer"]

def evaluate_answer_semantic(question, user_answer, context, threshold=0.4):
    extracted_answer = answer_question(question, context)
    vectorizer = TfidfVectorizer().fit_transform([user_answer, extracted_answer])
    similarity = cosine_similarity(vectorizer[0:1], vectorizer[1:2])[0][0]
    if similarity >= threshold:
        feedback = f"Good job! Your answer seems correct.\nDocument says: '{extracted_answer}'"
    else:
        feedback = f"Oops! Your answer seems incorrect.\nDocument suggests: '{extracted_answer}'"
    return feedback

# ----------------------------
# MAIN APP
# ----------------------------
st.title("Smart Assistant for Research Summarization")

uploaded_file = st.file_uploader("Upload your research paper (.pdf or .txt)", type=['pdf', 'txt'])

if uploaded_file:
    if "document_text" not in st.session_state:
        if uploaded_file.name.endswith(".pdf"):
            document_text = extract_text_from_pdf(uploaded_file)
        else:
            document_text = uploaded_file.read().decode("utf-8")
        st.session_state.document_text = document_text

    if show_preview:
        st.subheader("Document Preview")
        st.write(st.session_state.document_text[:1000] + "...")

    if st.button("Summarize", key="summarize_btn") or st.session_state.get("summary_done", False):
        if not st.session_state.get("summary_done", False):
            with st.spinner("Summarizing…"):
                summary = summarize_text(st.session_state.document_text)
                st.session_state.summary = summary
                st.session_state.summary_done = True

        st.subheader("Summary")
        st.write(st.session_state.summary)

        st.markdown("---")
        st.subheader("Available Options")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Generate Questions", key="questions_btn") or st.session_state.get("questions_done", False):
                if not st.session_state.get("questions_done", False):
                    with st.spinner("Generating & correcting questions…"):
                        questions = generate_questions(st.session_state.document_text)
                        corrected_questions = [correct_grammar(q) for q in questions]
                        st.session_state.questions = corrected_questions
                        st.session_state.questions_done = True

        with col2:
            if st.button("Challenge Me", key="challenge_btn") or st.session_state.get("challenge_me", False):
                st.session_state.challenge_me = True

        if st.session_state.get("questions_done", False):
            st.subheader("Auto-Generated Questions")
            for i, q in enumerate(st.session_state.questions, 1):
                st.write(f"*Q{i}:* {q}")

        if st.session_state.get("challenge_me", False):
            st.subheader("Challenge Me!")
            st.write("Answer the following questions and get instant feedback:")

            if not st.session_state.get("questions_done", False):
                st.info("⚠ Please generate questions first before attempting Challenge Me.")
            else:
                for i, q in enumerate(st.session_state.questions, 1):
                    user_answer = st.text_input(f"Your Answer for Q{i}: {q}", key=f"user_answer_{i}")
                    key_feedback = f"feedback_{i}"
                    if user_answer:
                        if key_feedback not in st.session_state:
                            with st.spinner("Evaluating…"):
                                feedback = evaluate_answer_semantic(q, user_answer, st.session_state.document_text)
                                st.session_state[key_feedback] = feedback
                        st.info(st.session_state[key_feedback])