import streamlit as st
import ollama
import os
from pypdf import PdfReader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from crewai import Agent, Task, Crew

# --- ১. পেজ কনফিগারেশন ও স্টাইল ---
st.set_page_config(page_title="AI Research Assistant", page_icon="📝", layout="wide")

# কাস্টম সিএসএস (UI সুন্দর করার জন্য)
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .stTextInput>div>div>input { border-radius: 5px; }
    </style>
    """, unsafe_allow_html=True)

# ফোল্ডার সেটআপ
if not os.path.exists("data"):
    os.makedirs("data")

# --- ২. সাইডবার ও হেডার ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
    st.title("Instruction")
    st.info("""
    ১. আপনার PDF পেপারটি আপলোড করুন।
    2. RAG সেকশনে নির্দিষ্ট প্রশ্ন করুন।
    ৩. AI এজেন্টদের মাধ্যমে গভীর বিশ্লেষণ (Agentic AI) শুরু করুন।
    """)

st.title("📝 Research Paper AI Assistant")
st.caption("Powered by Llama 3 & CrewAI | Advanced Paper Analysis System")

# --- ৩. ফাইল আপলোডার ---
uploaded_file = st.file_uploader("আপনার রিসার্চ পেপার (PDF) এখানে ড্রপ করুন", type="pdf")

if uploaded_file is not None:
    file_path = os.path.join("data", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # প্রসেসিং শুরু
    with st.spinner("পেপারটি বিশ্লেষণ করা হচ্ছে..."):
        # ৩.১ পিডিএফ থেকে টেক্সট বের করা
        reader = PdfReader(uploaded_file)
        paper_text = ""
        for page in reader.pages:
            paper_text += page.extract_text()

        # ৩.২ টেক্সট চাংকিং ও ভেক্টর ডাটাবেস (RAG - Project 2)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_text(paper_text)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # করাপশন এড়াতে persist_directory ছাড়াই রান করা হচ্ছে
        vector_db = Chroma.from_texts(chunks, embeddings)
        st.success(f"ফাইল '{uploaded_file.name}' সফলভাবে প্রসেস হয়েছে!")

    # --- লেআউট কলাম তৈরি (Project 2 & 3 পাশাপাশি বা নিচে নিচে) ---
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔍 Quick Insights (RAG)")
        query = st.text_input("পেপারটি সম্পর্কে প্রশ্ন করুন (যেমন: Methodology কি?):")
        
        if query:
            docs = vector_db.similarity_search(query, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            final_prompt = f"Answer based on context:\nContext: {context}\nQuestion: {query}"
            
            with st.chat_message("assistant"):
                response = ollama.generate(model='llama3', prompt=final_prompt)
                st.write(response['response'])

    with col2:
        st.subheader("🤖 Deep Analysis (Multi-Agent)")
        st.write("দুইজন বিশেষজ্ঞ এআই এজেন্ট আপনার পেপারটি রিভিউ করবে।")
        
        if st.button("Start Agent Discussion"):
            # এজেন্ট ডিফাইন করা
            research_analyst = Agent(
                role='Senior Research Analyst',
                goal='Extract precise methodology and data insights',
                backstory='Expert in CS research paper analysis with 10 years experience.',
                llm='ollama/llama3',
                verbose=True
            )

            critic_agent = Agent(
                role='Academic Critic',
                goal='Identify weaknesses and technical gaps',
                backstory='A skeptical professor specialized in identifying research limitations.',
                llm='ollama/llama3',
                verbose=True
            )

            # টাস্ক নির্ধারণ
            task1 = Task(
                description=f"Analyze this paper extract: {paper_text[:4000]}",
                agent=research_analyst,
                expected_output="A detailed summary of the research methodology and key findings."
            )

            task2 = Task(
                description="Critique the methodology found and list 3 major limitations.",
                agent=critic_agent,
                expected_output="A formal critique focusing on 3 specific limitations."
            )

            # ক্রু গঠন
            crew = Crew(agents=[research_analyst, critic_agent], tasks=[task1, task2])

            with st.status("এজেন্টরা আলোচনা করছে...", expanded=True) as status:
                st.write("Researcher is reading methodology...")
                result = crew.kickoff()
                status.update(label="বিশ্লেষণ সম্পন্ন!", state="complete", expanded=False)
            
            st.markdown("---")
            st.markdown("### 📋 Final Agent Report")
            st.info(result)

else:
    st.warning("শুরু করতে একটি পিডিএফ ফাইল আপলোড করুন।")