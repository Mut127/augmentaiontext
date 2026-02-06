import google.generativeai as genai

# === MASUKKAN API KEY KAMU DI SINI ===
API_KEY = "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" 

genai.configure(api_key=API_KEY)

print("Sedang menghubungi Google...")
print("Daftar model yang BISA dipakai:")
print("-" * 30)

try:
    available_models = []
    for m in genai.list_models():
        # Kita hanya cari model yang bisa generate text (generateContent)
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
            available_models.append(m.name)
            
    if not available_models:
        print("TIDAK ADA model yang ditemukan. Cek API Key atau koneksi internet.")
        
except Exception as e:
    print(f"Error saat cek model: {e}")