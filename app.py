import pandas as pd
from openai import OpenAI
import time
import json
import os
import sys

# ================= KONFIGURASI =================
# 1. Masukkan API Key dari OPENROUTER (Bukan Google AI Studio)
OPENROUTER_API_KEY = "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" 

# 2. Nama Model di OpenRouter (Yang gratis)
# Opsi lain: "google/gemini-2.0-pro-exp-02-05:free" (kalau ada)
MODEL_NAME = "google/gemini-2.0-flash-lite-001"

INPUT_FILE = "basis_pengetahuan_label_kurang_50_clean2.xlsx"
OUTPUT_FILE = "dataset_augmented_openrouter2.csv"

# Setup Client OpenAI tapi diarahkan ke OpenRouter
client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key=OPENROUTER_API_KEY,
)

def generate_via_openrouter(kode, deskripsi):
    prompt = f"""
    Bertindaklah sebagai pembuat dataset.
    Konteks: Kode KBLI {kode}. Definisi: {deskripsi}
    Tugas: Buatkan 50 variasi deskripsi pekerjaan SINGKAT (3-8 kata) sehari-hari.
    Format: JSON List murni. Contoh: ["saya berjual bakso", "memiliki perkebunan jagung""memiliki toko kelontong"]
    HANYA JSON, tanpa markdown.
    """

    # LOOP TAK TERBATAS (Sampai berhasil)
    while True:
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                extra_headers={
                    "HTTP-Referer": "https://localhost:3000", 
                    "X-Title": "DataGen", 
                }
            )
            
            content = completion.choices[0].message.content
            clean_content = content.replace("```json", "").replace("```", "").strip()
            data = json.loads(clean_content)
            
            if isinstance(data, list) and len(data) > 0:
                return data # BERHASIL! Keluar dari fungsi
                
        except Exception as e:
            # Ubah error jadi string biar bisa dicek
            error_msg = str(e)
            
            # Kalau errornya 429 (Limit), kita tunggu LAMA
            if "429" in error_msg:
                print(f"   ⏳ Lagi Macet (429). Nunggu 60 detik ya...", end="\r")
                time.sleep(60) # Tidur 1 menit
                print("   🔄 Mencoba lagi...                        ", end="\r")
            else:
                # Kalau error lain, tunggu sebentar aja
                print(f"   ⚠️ Error: {error_msg}. Coba lagi 10 detik...", end="\r")
                time.sleep(10)

def main():
    # 1. BACA FILE
    print("📂 Membaca file input...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except:
        try:
            df = pd.read_excel(INPUT_FILE.replace('.csv', ''))
        except:
            print("❌ File tidak ditemukan!")
            return

    # 2. PILIH RANGE (Supaya fleksibel)
    print(f"Total data: {len(df)}")
    try:
        start_idx = int(input("Mulai dari baris ke berapa? (0): ") or 0)
        limit = int(input("Mau berapa data? (10): ") or 10)
    except:
        start_idx = 0
        limit = 10
    
    df_slice = df.iloc[start_idx : start_idx + limit]
    
    print(f"\n🚀 Memulai proses via OpenRouter ({MODEL_NAME})...")
    
    for index, row in df_slice.iterrows():
        kode = row['kode']
        deskripsi = row['deskripsi']
        
        print(f"[{index}] Kode {kode}...", end=" ", flush=True)
        
        hasil = generate_via_openrouter(kode, deskripsi)
        
        if hasil:
            print(f"✅ OK ({len(hasil)} item)")
            
            # Simpan
            temp_df = pd.DataFrame([{'kode': kode, 'text': t} for t in hasil])
            hdr = not os.path.exists(OUTPUT_FILE)
            temp_df.to_csv(OUTPUT_FILE, mode='a', header=hdr, index=False)
            
            # Jeda sopan (OpenRouter free juga punya limit, tapi lebih santai)
            time.sleep(3) 
        else:
            print("❌ GAGAL")
            time.sleep(2)

    print("\n🎉 Selesai!")

if __name__ == "__main__":
    main()