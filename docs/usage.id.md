# Panduan penggunaan

[English version](usage.md) | [Gambaran proyek](project.md) | [Troubleshooting](troubleshooting.md)

Jalankan perintah terminal dari folder utama proyek, kecuali ada keterangan lain.

## 1. Tentukan yang ingin dijalankan

- **Lihat hasil:** buka README atau `results/reference/metrics.csv`.
- **Cek repository dan tes:** cukup Python 3.10+, tanpa GPU atau download model.
- **Buat ulang grafik:** pasang `requirements-report.txt`.
- **Jalankan notebook model:** perlu PyTorch, dependency proyek, data lokal, akses model, dan memori yang cukup.
- **Pakai model lewat Ollama:** perlu file GGUF hasil konversi dan Ollama yang sudah berjalan.

Cek ringan:

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
```

## 2. Siapkan environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Pasang PyTorch yang sesuai dengan OS dan hardware terlebih dahulu. Setelah itu:

```bash
python -m pip install -r requirements.txt
python -m ipykernel install --user --name llama32-1b-llm-abliteration --display-name "Llama 3.2 1B - LLM Abliteration"
```

Buka folder proyek di VS Code, lalu pilih kernel **Llama 3.2 1B - LLM Abliteration** pada notebook. Notebook 00 hanya mengecek versi package dan ketersediaan CUDA.

Environment awal memakai Python 3.10. `requirements.txt` berisi daftar dependency, bukan salinan versi environment lama yang dikunci. Kompatibilitas runtime model tetap bergantung pada versi yang dipasang.

## 3. Isi konfigurasi

Salin file contoh:

```powershell
Copy-Item .env.example .env
```

Di Linux/macOS, gunakan `cp .env.example .env`.

| Variabel | Kegunaan |
| --- | --- |
| `HF_TOKEN` | Token akses model bila diperlukan; kosongkan jika tidak perlu |
| `DEFAULT_MODEL` | Default `meta-llama/Llama-3.2-1B-Instruct` |
| `SEED` | Default `42` |
| `OLLAMA_HOST` | Default `http://localhost:11434` |
| `OLLAMA_MODEL` | Default `llama3.2-1b:abliteration` |
| `COLAB_DRIVE_PATH` | Path proyek di Drive, bila diatur sebagai variabel environment Colab |

Untuk mengikuti alur yang tersedia, pakai model default. Mengganti nama model saja belum cukup untuk mendukung arsitektur lain karena notebook konversi HF ditulis untuk Llama yang dipakai di proyek ini.

Restart kernel setelah mengubah konfigurasi. Environment variable dari terminal lebih diprioritaskan daripada `.env`. Khusus Colab, path proyek dicari sebelum `.env` dibaca: ubah sel `PROJECT_DIR` atau atur `COLAB_DRIVE_PATH` langsung di environment Colab.

Batch size dan batas token ada di `src/config.py`. Beberapa sel notebook punya nilai langsung, jadi periksa juga sel terkait ketika mengganti pengaturan.

## 4. Siapkan CSV

Format lengkap ada di [data/README.md](../data/README.md):

```text
data/harmful_prompts.csv   kolom prompt
data/harmless_prompts.csv  kolom prompt
data/eval_holdout.csv      kolom prompt,type
```

Kolom `kategori` boleh ditambahkan. Gunakan UTF-8 dan hindari prompt kosong. Contoh satu baris harmless:

```csv
prompt,kategori
"Explain how a rainbow forms.",Science
```

Dua CSV utama wajib ada. Kalau holdout belum ada, notebook 01 membuatnya melalui loader. Dengan pengaturan default, siapkan minimal 436 baris unik per kelas untuk alokasi 100 holdout, 256 train, dan 80 test. Data lebih sedikit bisa membuat potongan test lebih pendek. Jumlah ini sendiri tidak mengatasi overlap pada alur ekstraksi; baca [catatan hasil](results.md).

Tiga CSV dataset eksperimen disertakan di `data/`. Memakai data sendiri akan menghasilkan angka yang berbeda. Loader tidak mengunduh dataset pengganti secara otomatis.

## 5. Jalankan notebook berurutan

Gunakan kernel baru untuk setiap notebook agar memori dan variabel dari langkah sebelumnya tidak terbawa.

| Urutan | Notebook | Output utama |
| --- | --- | --- |
| 00 | `00_setup_torch.ipynb` | Informasi environment |
| 01 | `01_baseline_eval.ipynb` | Log baseline dan tabel metrik lokal |
| 02 | `02_extract_activations.ipynb` | Tensor arah representasi |
| 03 | `03_screening_directions.ipynb` | CSV screening |
| 04 | `04_orthogonalize_save.ipynb` | Checkpoint modifikasi |
| 05 | `05_convert_to_hf.ipynb` | Model dan tokenizer HF |
| 06 | `06_eval_abliterated.ipynb` | Metrik model HF |
| 07 | `07_convert_gguf.ipynb` | GGUF F16 |
| 08 | `08_ollama_eval.ipynb` | Metrik respons Ollama |

Untuk tahap 02, tersedia alternatif `02_extract_activations_colab.ipynb`. Pilih salah satu, bukan menjalankan keduanya berurutan. Salin proyek dan CSV lokal ke Drive, mount Drive, lalu sesuaikan `PROJECT_DIR`. Kalau melanjutkan di komputer lokal, salin artefak yang dihasilkan ke lokasi folder yang sama.

Setup awal memakai RTX 2050 4 GB dan Colab untuk ekstraksi. Kebutuhan memori berubah mengikuti panjang prompt, dtype, dan versi package. Jika kehabisan memori, lihat [troubleshooting](troubleshooting.md).

## 6. Konversi GGUF dan jalankan Ollama

Notebook 07 mencari checkout llama.cpp di `~/llama.cpp`. Pastikan `convert_hf_to_gguf.py` dan dependency converter sudah tersedia. Sesuaikan `LLAMA_CPP_DIR` jika lokasinya berbeda. File outputnya adalah `models/gguf/model.gguf`, berformat F16.

Setelah Ollama terpasang dan server lokalnya berjalan, jalankan dari root proyek:

```bash
ollama create llama3.2-1b:abliteration -f Modelfile
ollama run llama3.2-1b:abliteration
```

Coba prompt sederhana, misalnya `Explain how a rainbow forms.`. Keluar dengan `/bye`, lalu gunakan notebook 08 untuk evaluasi dari CSV. Nama model harus sama dengan `OLLAMA_MODEL` di `.env`.

Notebook 08 melewati import jika tag sudah terdaftar. Kalau GGUF diganti, jalankan lagi `ollama create` agar model lokal diperbarui. Perintah ini tidak mengunggah model ke registry.

## 7. Baca hasilnya

- `results/metrics.csv`: metrik run lokal. Notebook 01 memulai ulang tabel ini; notebook berikutnya menambah baris.
- `results/generations_log/`: respons mentah lokal.
- `results/reference/`: hasil contoh yang ikut repository; tidak otomatis ditimpa notebook.

Menjalankan sel simpan berkali-kali bisa menambah baris duplikat. Salin hasil lokal yang ingin dipertahankan sebelum mengulang alur.

Untuk membuat ulang grafik referensi:

```bash
python -m pip install -r requirements-report.txt
python scripts/plot_results.py
```

Script grafik membaca CSV referensi, bukan `results/metrics.csv` dari run terbaru. Batas perbandingan angka dijelaskan di [catatan hasil](results.md).

## 8. Siapkan file untuk dibagikan

Bersihkan output notebook sebelum menyimpan: di VS Code, gunakan **Clear All Outputs**. Execution counter juga harus kosong; cek hasilnya dengan script berikut.

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
python scripts/package_repo.py
```

Paket tersedia di `dist/llama32-1b-llm-abliteration.zip`. Isinya hanya kode, dokumentasi, dan hasil referensi yang sudah dicek. Tiga CSV prompt eksperimen ikut disertakan. Model, token, dan respons mentah tidak ikut. Panduan push tersedia di [setup GitHub](github.md).
