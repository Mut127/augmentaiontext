from pydoc import text
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from symspellpy import SymSpell, Verbosity
import torch
import joblib
import re
import mysql.connector
import requests

last_business_text = {}

app = Flask(__name__)
CORS(app)

# ===============================
# Load Model & Tokenizer IndoBERT
# ===============================
model_path = "model_chatkbli2020"
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load label encoder
le = joblib.load("label_encoder.pkl")

# ===============================
# SymSpell Initialization
# ===============================
sym_spell = SymSpell(
    max_dictionary_edit_distance=2,
    prefix_length=7
)

# ===============================
# Database
# ===============================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="descan"
)
cursor = db.cursor(dictionary=True)

# ===============================
# OpenRouter CONFIG
# ===============================
USE_LLM = True   # False = full offline
OPENROUTER_API_KEY = "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"
OPENROUTER_MODEL = "google/gemini-2.0-flash-lite-001"

from flask import Flask, render_template
import mysql.connector  # atau SQLAlchemy


def get_kbli_categories():
    cursor.execute("""
        SELECT DISTINCT kategori AS kode, nama_kategori AS judul
        FROM kbli_2020
        WHERE kategori IS NOT NULL AND kategori != ''
        ORDER BY kategori
    """)
    rows = cursor.fetchall()  # list of dict

    # Trim spasi & filter
    categories = [
        {"kode": row['kode'].strip(), "judul": row['judul'].strip()} 
        for row in rows 
        if row['kode'] and row['kode'].strip() != ''
    ]

    # Debug console
    print(f"{'Kode':<5} {'Kategori'}")
    print("-"*40)
    for cat in categories:
        print(f"{cat['kode']:<5} {cat['judul']}")

    return categories



# Panggil fungsi
get_kbli_categories()

# ===============================
# Build SymSpell Dictionary
# ===============================
def build_symspell_dictionary():
    cursor.execute("SELECT judul, deskripsi FROM kbli_2020")
    rows = cursor.fetchall()

    for row in rows:
        text = f"{row['judul']} {row['deskripsi']}".lower()
        words = re.findall(r"[a-zA-Z]+", text)
        for w in words:
            sym_spell.create_dictionary_entry(w, 1)

build_symspell_dictionary()

# ===============================
# Helper Functions
# ===============================
def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def correct_typo(text):
    corrected = []
    for word in text.split():
        suggestions = sym_spell.lookup(
            word,
            Verbosity.CLOSEST,
            max_edit_distance=2
        )
        corrected.append(suggestions[0].term if suggestions else word)
    return " ".join(corrected)

FOOD_KEYWORDS = {
    "cilok", "bakso", "jualan", "berjualan", "makanan",
    "jajanan", "minuman", "warung", "kaki", "lima",
    "gerobak", "kuliner", "dagangan"
}

SYSTEM_PROMPT = """
    Anda adalah asisten UMKM bernama BAKUL KAHURIPAN.

    TOPIK UTAMA:
    - UMKM
    - Jenis usaha
    - Perizinan usaha
    - KBLI
    - NIB
    - Sertifikasi halal

    ATURAN:
    - Jawab dengan bahasa Indonesia yang ramah
    - Jika pertanyaan di luar topik UMKM, arahkan kembali ke topik UMKM
    - Jika pengguna belum menjelaskan jenis usaha, mintalah penjelasan usaha
    - Jangan menentukan KBLI tanpa data dari sistem
    - Jangan menebak atau mengarang jenis usaha

    JIKA:
    - user sudah menyebut jenis usaha
    - DAN model klasifikasi confidence ≥ threshold

    MAKA:
    - hentikan pertanyaan
    - langsung tampilkan KBLI

    PERAN ANDA:
    - Menjawab pertanyaan umum seputar UMKM
    - Mengarahkan pengguna agar menjelaskan usaha
    - Menjelaskan hasil KBLI yang diberikan sistem
    """

def has_food_activity(text):
    words = set(text.split())
    return bool(words & FOOD_KEYWORDS)

def is_non_business_kbli(judul):
    blacklist = ["PENDIDIKAN", "GEDUNG", "SEKOLAH", "PEMERINTAH"]
    return any(b in judul.upper() for b in blacklist)


def keyword_relevance(query, description):
    score = 0
    desc = description.lower()
    query_words = set(query.split())

    for q in query_words:
        if q in desc:
            score += 2
        elif any(q in d for d in desc.split()):
            score += 1

    return score
def call_openrouter(messages):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost",
        "X-Title": "Chatbot KBLI Skripsi"
    }

    payload = {
        "model": OPENROUTER_MODEL,
        "messages": messages,
        "temperature": 0.3
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=payload,
        timeout=30
    )

    return response.json()["choices"][0]["message"]["content"]


def generate_chat_response(user_text, best_kbli):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
    Berikut adalah hasil klasifikasi sistem:

    Kode KBLI: {best_kbli['kode']}
    Judul: {best_kbli['judul']}
    Deskripsi:
    {best_kbli['deskripsi']}

    Tolong jelaskan hasil ini dengan bahasa sederhana dan ramah.
            """}
        ]

    return call_openrouter(messages)


# ===============================
# Rule Stop
# ===============================

MAX_CLARIFICATION = 2
CONFIDENCE_THRESHOLD = 0.15

def model_confidence(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    return torch.max(probs).item()

def is_direct_kbli_request(text):
    keywords = ["kbli", "kode kbli", "berapa kbli", "kbli apa"]
    text = text.lower()
    return any(k in text for k in keywords)

user_clarification_count = {}

def should_stop_and_classify(user_text, session_id):
    # Init counter
    if session_id not in user_clarification_count:
        user_clarification_count[session_id] = 0

    # 1️⃣ User minta KBLI langsung
    if is_direct_kbli_request(user_text):
        return True

    # 2️⃣ Confidence model cukup
    confidence = model_confidence(user_text)
    if confidence >= CONFIDENCE_THRESHOLD:
        return True

    # 3️⃣ Klarifikasi sudah maksimal
    if user_clarification_count[session_id] >= MAX_CLARIFICATION:
        return True

    return False

# ===============================
# Routes
# ===============================
@app.route("/")
def home():
    kbli_categories = get_kbli_categories()
    return render_template("index.html", kbli_categories=kbli_categories)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        text = data.get("use_text") or data.get("text", "")

        # ===============================
        # VALIDASI INPUT
        # ===============================
        if not text or text.strip() == "":
            return jsonify({
                "success": False,
                "error": "Teks input kosong"
            })

        text = normalize_text(text)
        text = correct_typo(text)

        if len(text.split()) < 3:
            return jsonify({
                "success": False,
                "error": "Deskripsi usaha terlalu singkat. Mohon jelaskan lebih detail."
            })

        # ===============================
        # TOKENISASI & INFERENSI
        # ===============================
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits, dim=1)[0]

        # ===============================
        # TOP-K PREDICTION
        # ===============================
        TOP_K = 20
        topk = torch.topk(probs, k=TOP_K)

        pred_ids = topk.indices.tolist()
        scores = topk.values.tolist()

        kode_list = [
            le.inverse_transform([pid])[0].zfill(5)
            for pid in pred_ids
        ]

        # ===============================
        # AMBIL DATA KBLI
        # ===============================
        format_strings = ",".join(["%s"] * len(kode_list))
        cursor.execute(
            f"""
            SELECT kode, judul, deskripsi
            FROM kbli_2020
            WHERE kode IN ({format_strings})
            """,
            tuple(kode_list)
        )

        rows = cursor.fetchall()
        db_map = {row["kode"]: row for row in rows}

        # ===============================
        # SUSUN HASIL
        # ===============================
        results = []
        is_food = has_food_activity(text)

        for i, kode in enumerate(kode_list):
            if kode not in db_map:
                continue

            relevance = keyword_relevance(
                text,
                db_map[kode]["deskripsi"]
            )

            judul = db_map[kode]["judul"].upper()
            deskripsi = db_map[kode]["deskripsi"].lower()

            # Penalti KBLI non-usaha
            if is_non_business_kbli(judul):
                relevance -= 3

            # BOOST KHUSUS MAKANAN
            if is_food:
                if any(k in deskripsi for k in ["makanan", "minuman", "restoran", "warung"]):
                    relevance += 6
                else:
                    relevance -= 2  # non-makanan diturunin

            results.append({
                "kode": kode,
                "judul": db_map[kode]["judul"],
                "deskripsi": db_map[kode]["deskripsi"],
                "score": round(float(scores[i]), 4),
                "relevance": relevance
            })


        # ===============================
        # SORTING UTAMA (RELEVANCE > SCORE)
        # ===============================
        results = sorted(
            results,
            key=lambda x: (x["relevance"], x["score"]),
            reverse=True
        )

        # ===============================
        # FILTER MINIMUM
        # ===============================
        MIN_RELEVANCE = 1
        MIN_SCORE = 0.05

        filtered = [
            r for r in results
            if r["relevance"] >= MIN_RELEVANCE and r["score"] >= MIN_SCORE
        ]

        if not filtered:
            filtered = results[:3]
        
        best_kbli = filtered[0]
        chat_reply = generate_chat_response(text, best_kbli)


        return jsonify({
            "success": True,
            "input": text,
            "recommendations": filtered,
            "chatbot_reply": chat_reply
        })

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        })
    
# @app.route("/chat", methods=["POST"])
# def chat():
#     data = request.get_json()
#     user_text = data.get("text", "").strip()

#     messages = [
#         {"role": "system", "content": SYSTEM_PROMPT},
#         {"role": "user", "content": user_text}
#     ]

#     reply = call_openrouter(messages)
#     return jsonify({"reply": reply})
@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_text = data.get("text", "").strip()
    session_id = data.get("session_id", "default")

    if not user_text:
        return jsonify({"reply": "Halo 👋 Ada yang bisa saya bantu?"})

    # 🔥 STOP RULE
    if should_stop_and_classify(user_text, session_id):
        return jsonify({"redirect": "predict"})

    # ⬆️ Kalau belum stop → klarifikasi
    user_clarification_count[session_id] += 1

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text}
    ]

    reply = call_openrouter(messages)
    return jsonify({"reply": reply})

def is_ready_for_classification_with_model(text, threshold=0.15):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]
    top_prob = torch.max(probs).item()

    return top_prob >= threshold


# ===============================
# Run Server
# ===============================
if __name__ == "__main__":
    app.run(debug=True)