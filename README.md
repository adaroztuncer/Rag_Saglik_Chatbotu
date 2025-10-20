# 🏥 RAG Tabanlı Türkçe Sağlık Asistanı Chatbotu

Bu proje, Türkçe sağlık sorularına doğru ve bağlama uygun yanıtlar sağlamak için **Retrieval-Augmented Generation (RAG)** tabanlı bir chatbot uygulamasıdır. Türk hastanelerinden toplanan tıbbi makaleler üzerine eğitilmiş olup, **LangChain**, **Chroma**, **Google Generative AI (Gemini API)** ve **Gradio** teknolojilerini entegre eder. Kullanıcı dostu bir web arayüzü üzerinden erişilebilir ve Hugging Face Spaces üzerinde barındırılabilir.

## 📜 Proje Amacı

Bu proje, sağlıkla ilgili Türkçe sorulara güvenilir ve bilgilendirici yanıtlar sunan bir sağlık asistanı chatbotu geliştirmeyi amaçlar. RAG mimarisi ile vektör tabanlı bilgi erişimi ve üretken yapay zeka modelini birleştirerek, kullanıcıların tıbbi bilgilere hızlı ve doğru bir şekilde erişmesini sağlar. Ayrıca, modern yapay zeka tekniklerinin sağlık alanındaki uygulamalarını göstermeyi hedefler.

---

## 🚀 Özellikler

- 📚 **RAG Tabanlı Yapay Zeka**: Yanıtlar, gerçek metin parçalarına dayalı olarak üretilir.
- 🔍 **Vektör Arama (ChromaDB)**: Kullanıcı sorularına en uygun metin parçalarını hızlıca bulur.
- 🤖 **Google Gemini API Entegrasyonu**: Doğal, tutarlı ve Türkçe odaklı yanıtlar üretir.
- 🧩 **LangChain Bileşenleri**: Modüler metin işleme ve embedding oluşturma.
- 💬 **Gradio Arayüzü**: Kullanıcı dostu, web tabanlı arayüz.
- ⚡ **Hugging Face Entegrasyonu**: Model barındırma ve paylaşım desteği.

---

## 🧱 Çözüm Mimarisi

Proje, aşağıdaki akışla çalışır:

```
Kullanıcı → Gradio UI → LangChain Pipeline
              ↓
          ChromaDB (Vektör Arama)
              ↓
   İlgili Metin Parçaları → Gemini API (RAG Cevap)
              ↓
          Türkçe Yanıt
```

1. **Kullanıcı Sorgusu**: Kullanıcı, Gradio arayüzü üzerinden Türkçe bir sağlık sorusu girer.
2. **Vektör Arama**: LangChain, sorguyu `sentence-transformers` ile vektörleştirir ve ChromaDB'den en alakalı metin parçalarını çeker.
3. **Yanıt Üretimi**: Çekilen metin parçaları, Google Gemini API'ye bağlam olarak gönderilir ve doğal bir Türkçe yanıt üretilir.
4. **Sonuç**: Yanıt, Gradio arayüzünde kullanıcıya sunulur.

---

## 🧰 Kullanılan Teknolojiler

| Teknoloji                 | Açıklama                        |
|---------------------------|-------------------------------|
| **Python 3.10+**          | Geliştirme dili                 |
| **LangChain**             | RAG pipeline ve metin işleme    |
| **Chroma**                | Vektör veritabanı               |
| **Sentence Transformers** | Metin vektörleştirme            |
| **Google Gemini API**     | Doğal dil yanıt üretimi         |
| **Gradio**                | Web tabanlı kullanıcı arayüzü   |
| **dotenv**                | Ortam değişkenleri yönetimi     |
| **Hugging Face Hub**      | Model barındırma (isteğe bağlı) |

---

## 📊 Veri Seti Hazırlama

### Veri Seti Kaynağı
Proje, Hugging Face üzerindeki `umutertugrul/turkish-hospital-medical-articles` veri setini kullanır. Bu veri seti, Acıbadem, Anadolu Sağlık, Memorial gibi Türk hastanelerinin web sitelerinden toplanan tıbbi makalelerden oluşur. Acıbadem Parquet dosyası temel alınmıştır.

- **Veri Seti Özellikleri**:
  - **Kayıt Sayısı**: 6,339 makale
  - **Alanlar**: `title`, `headings`, `text`, `url`, `publish_date`, `update_date`, `scrape_date`, `__source`
  - **İçerik**: Başlık, alt başlıklar ve makale metinleri birleştirilerek `content` sütunu oluşturulmuştur.
  - **Kaynak**: Türk hastanelerinin web siteleri (özellikle Acıbadem)

### Veri Ön İşleme
1. Veri seti, Hugging Face `datasets` kütüphanesi ile yüklenip Pandas DataFrame'e dönüştürülmüştür.
2. `title`, `headings` ve `text` sütunları birleştirilerek `content` sütunu oluşturulmuştur.
3. Metinler, `RecursiveCharacterTextSplitter` ile ~1000 karakterlik parçalara ayrılmış, 100 karakterlik örtüşme sağlanmıştır.
4. Toplam **38,544** metin parçası elde edilmiştir.
5. Parçalar, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` modeli ile vektörleştirilmiş ve Chroma veritabanında saklanmıştır.

---

## ⚙️ Çalışma Kılavuzu

### Gereksinimler
- **Python**: 3.10 veya üstü
- **API Anahtarları**: Google Gemini API ve (isteğe bağlı) Hugging Face Hub API anahtarı
- **Kütüphaneler**: `requirements.txt` dosyasında listelenmiştir

### Kurulum Adımları
1. **Depoyu Klonlayın**:
   ```bash
   git clone https://github.com/adaroztuncer/Rag_Saglik_Chatbotu.git
   cd Rag_Saglik_Chatbotu
   ```

2. **Sanal Ortam Oluşturun ve Aktif Edin**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. **Gerekli Kütüphaneleri Yükleyin**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Ortam Değişkenlerini Ayarlayın**:
   `.env` dosyasını oluşturun ve API anahtarlarınızı ekleyin:
   ```plaintext
   GEMINI_API_KEY=your_google_gemini_api_key
   HF_TOKEN=your_huggingface_token
   ```

5. **Veri Setini Yükleyin**:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("umutertugrul/turkish-hospital-medical-articles")
   ```

6. **Vektör Veritabanını Oluşturun**:
   ```bash
   python build_index.py
   ```

7. **Uygulamayı Başlatın**:
   ```bash
   python app.py
   ```

### Google Colab'da Çalıştırma
1. `.ipynb` dosyasını Google Colab'e yükleyin.
2. API anahtarlarınızı Colab'in "Secrets" bölümüne ekleyin.
3. Google Drive'ınızı bağlayarak veri ve vektör veritabanını kaydedin.
4. Notebook hücrelerini sırayla çalıştırın.

### Hugging Face Spaces'de Çalıştırma
1. `app.py` ve `requirements.txt` dosyalarını Hugging Face Space'e yükleyin.
2. API anahtarlarınızı Space'in "Secrets" bölümüne ekleyin.
3. Space'i başlatın.

---

## 💻 Web Arayüzü & Product Kılavuzu

### Web Arayüzü
- **Teknoloji**: Gradio
- **Özellikler**:
  - Kullanıcılar, sağlıkla ilgili Türkçe sorularını metin kutusuna yazabilir.
  - Chatbot, ChromaDB'den ilgili metin parçalarını çeker ve Gemini API ile doğal yanıtlar üretir.
  - Arayüz, basit ve kullanıcı dostudur; başlık, açıklama ve metin kutusu içerir.
- **Örnek Kullanım**:
  - **Soru**: "Hemoglobin nedir?"
  - **Yanıt**: "Hemoglobin, kırmızı kan hücrelerinde bulunan ve oksijen taşıyan bir proteindir. Vücudun dokularına oksijen taşınmasında kritik bir rol oynar..."

### Ürün Kılavuzu
- **Hedef Kullanıcılar**: Sağlık bilgisi arayan bireyler, sağlık profesyonelleri, öğrenciler
- **Kullanım Senaryoları**:
  - Tıbbi terimlerin açıklamalarını öğrenme
  - Semptomlar hakkında genel bilgi alma
  - Sağlık makalelerine dayalı öneriler
- **Kısıtlamalar**:
  - Yanıtlar, veri setindeki bilgilere sınırlıdır.
  - Tıbbi teşhis veya tedavi önerisi sunmaz; profesyonel sağlık hizmetlerinin yerini almaz.
- **Erişim**: Web arayüzü, yerel ortamda veya Hugging Face Spaces üzerinden kullanılabilir.

---

## 📦 Geliştirme Ortamı

- **GitHub**: Proje, [GitHub deposunda](https://github.com/adaroztuncer/Rag_Saglik_Chatbotu) barındırılmaktadır. Kod, dokümantasyon ve veri işleme script'leri düzenli bir şekilde organize edilmiştir.
- **README.md**: Projenin amacı, kurulum adımları, kullanım kılavuzu ve teknik detayları içerir.
- **Hugging Face Space**: Proje, [Hugging Face Space](https://huggingface.co/spaces/adaroztuncer/health-chatbot) üzerinden erişilebilir ve çalıştırılabilir.
- **Dosya Yapısı**:
  ```
  Rag_Saglik_Chatbotu/
  ├── app.py                     # Gradio arayüzü ve chatbot mantığı (Huggingface için)
  ├── rag_chatbot.ipynb          # Veri yükleme ve vektör oluşturma (Notebook colalb için)
  ├── requirements.txt           # Gerekli kütüphaneler
  ├── .env                       # API anahtarları (sembolik)
  ├── README.md                  # Proje dokümantasyonu
  └── images/                    # Proje görselleri
  ```

---

## 📈 Elde Edilen Sonuçlar

- **Veri İşleme**: 6,339 makale başarıyla işlenmiş ve 38,544 metin parçasına ayrılmıştır.
- **Vektör Veritabanı**: ChromaDB, hızlı ve doğru bilgi erişimi için 38,544 vektör saklamaktadır.
- **Chatbot Performansı**: Türkçe sağlık sorularına bağlama uygun, doğru ve bilgilendirici yanıtlar üretir.
- **Kullanıcı Deneyimi**: Gradio arayüzü, kullanıcıların kolayca etkileşime geçmesini sağlar.

---

## 🧠 Geliştirici Notları

- `app.py` içindeki **MODEL_CONFIG** ayarları, yanıt kalitesini optimize etmek için kullanılabilir.
- Hugging Face Spaces'de, `app_file` olarak `app.py` belirtilmelidir.

---

## 🔮Deploy Linki
https://huggingface.co/spaces/adaroztuncer/health-chatbot


---



```
MIT License © 2025 Adar Öztuncer
```

