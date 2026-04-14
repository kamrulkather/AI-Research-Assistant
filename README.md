# 📝 Research Paper AI Assistant

এটি একটি উন্নত AI-চালিত টুল যা **Streamlit**, **LangChain**, এবং **CrewAI** ব্যবহার করে যেকোনো PDF রিসার্চ পেপার বিশ্লেষণ করতে পারে।

## ✨ মূল বৈশিষ্ট্য (Features)
- **RAG System (Project 2):** PDF থেকে তথ্য নিয়ে আপনার করা যেকোনো প্রশ্নের সঠিক উত্তর দেয়।
- **Multi-Agent System (Project 3):** দুইজন AI বিশেষজ্ঞ (Researcher & Critic) আপনার পেপারটি নিয়ে আলোচনা এবং সমালোচনা করে।
- **Privacy Focused:** এটি লোকাল LLM (Ollama - Llama 3) ব্যবহার করে, ফলে আপনার ডাটা আপনার পিসিতেই থাকে।

## 🛠️ প্রযুক্তি (Tech Stack)
- **Frontend:** Streamlit
- **Orchestration:** LangChain & CrewAI
- **Embedding:** HuggingFace (all-MiniLM-L6-v2)
- **Model:** Llama 3 (via Ollama)

## 🚀 কীভাবে রান করবেন?
১. আপনার পিসিতে Ollama এবং Llama 3 ইনস্টল থাকতে হবে।
২. ভার্চুয়াল এনভায়রনমেন্ট অ্যাক্টিভেট করুন।
৩. প্রয়োজনীয় লাইব্রেরি ইনস্টল করুন:
   ```bash
   pip install -r src/requirements.txt