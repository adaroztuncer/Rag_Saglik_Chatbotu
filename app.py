# -*- coding: utf-8 -*-
import os
import pandas as pd
import gradio as gr
import google.generativeai as genai
from huggingface_hub import login
from datasets import load_dataset
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Ortam değişkenlerinden API anahtarlarını al
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

# Google Generative AI yapılandırması
genai.configure(api_key=GEMINI_API_KEY)

# Hugging Face'e giriş yap
try:
    login(token=HF_TOKEN)
    print("✅ Hugging Face login başarılı.")
except Exception as e:
    print(f"❌ Giriş başarısız: {e}")

# Veri setini yükle (sadece acibadem parçasını)
print("📦 Sadece 'acibadem' veri seti yükleniyor...")
data_split = load_dataset(
    "umutertugrul/turkish-hospital-medical-articles",
    split='acibadem'
)

# Veri setinin sütunlarını kontrol et
print("Veri seti sütunları:", data_split.column_names)

# DataFrame'e dönüştür
print("Veri DataFrame'e dönüştürülüyor...")
df = data_split.to_pandas()

# Varsayılan olarak 'title' ve 'text' sütunlarını kullan
available_columns = [col for col in ['title', 'text'] if col in df.columns]
if not available_columns:
    raise ValueError("Gerekli sütunlar ('title' veya 'text') veri setinde bulunamadı!")

# Content sütununu oluştur
df['content'] = df.apply(
    lambda row: f"{row['title'] if 'title' in row else ''}\n\n{row['text'] if 'text' in row else ''}",
    axis=1
)

print(f"✅ Toplam kayıt (sadece acibadem): {len(df)}")

# Metinleri parçalara ayır
print("🔹 Metinler parçalara ayrılıyor...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=50,
    separators=["\n\n", "\n", ".", " "]
)
chunks = []
for text in df['content'].tolist():
    chunks.extend(text_splitter.split_text(text))
print(f"✅ Toplam chunk sayısı: {len(chunks)}")

# SentenceTransformer Embeddings sınıfı
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_tensor=False).tolist()
    def embed_query(self, text):
        return self.model.encode([text], convert_to_tensor=False)[0].tolist()

# Embedding modeli
embedding_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Chroma vektör veritabanını oluştur veya yükle
persist_directory = "./chroma_index"
collection_name = "turhish_hospital_articles"

if os.path.exists(persist_directory):
    print("✅ Mevcut Chroma vektör veritabanı yükleniyor...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
else:
    print("🔹 Yeni Chroma vektör veritabanı oluşturuluyor...")
    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    print("✅ Chroma vektör veritabanı oluşturuldu ve kaydedildi.")
print(f"📦 Toplam vektör sayısı: {db._collection.count()}")

# LLM (Gemini Flash)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.6,
    max_tokens=600,
    google_api_key=GEMINI_API_KEY
)

# Chroma retriever
retriever = db.as_retriever(search_kwargs={"k": 3})

# Prompt şablonu
system_prompt = (
    "Sen bir sağlık asistanısın. Kullanıcıdan gelen soruları, "
    "Chroma veritabanından getirilen ilgili metin parçalarına dayanarak yanıtla. "
    "Eğer soruyu yanıtlamak için yeterli bilgi yoksa, 'Bu konu hakkında yeterli bilgiye sahip değilim.' de. Daha fazla bilgi iste.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}")
    ]
)

# RAG zinciri
chain = create_retrieval_chain(
    retriever=retriever,
    combine_docs_chain=create_stuff_documents_chain(llm=llm, prompt=prompt)
)

# Gradio fonksiyonu
def answer_question(user_input, history):
    try:
        # Önceki konuşmaları birleştirip modele bağlam olarak gönder
        context = "\n".join([f"Kullanıcı: {q}\nAsistan: {a}" for q, a in history])
        full_input = f"{context}\nKullanıcı: {user_input}\nAsistan:"
        
        # Modeli çağır (örnek olarak chain.invoke)
        response = chain.invoke({"input": full_input})["answer"]

        # Yeni mesajı geçmişe ekle
        history = history + [[user_input, response]]
        return "", history  # giriş kutusunu temizle
    except Exception as e:
        return "", history + [[user_input, f"Bir hata oluştu: {e}"]]

# Gradio arayüzü
with gr.Blocks(theme="soft") as demo:
    gr.Markdown("## 🏥 Sağlık Asistanı Chatbot")
    gr.Markdown(
        "Sağlıkla ilgili sorularınızı yanıtlayan chatbot. "
        "Hastalık belirtileri, tedaviler ve daha fazlası hakkında bilgi alabilirsiniz."
    )

    chatbot = gr.Chatbot(label="💬 Sohbet Alanı")
    user_input = gr.Textbox(
        lines=2,
        placeholder="Sorunuzu buraya yazın ve Enter’a basın...",
        label="Soru",
    )
    submit_btn = gr.Button("Gönder")

    # Butonla gönderme
    submit_btn.click(
        fn=answer_question,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )

    # Enter ile gönderme
    user_input.submit(
        fn=answer_question,
        inputs=[user_input, chatbot],
        outputs=[user_input, chatbot]
    )

# Uygulamayı başlat
if __name__ == "__main__":
    demo.launch()


