# Llama 3.2 1B - LLM Abliteration

Proyek notebook Python untuk mencoba abliteration pada Llama 3.2 1B, membandingkan responsnya, lalu menjalankan model hasil konversi lewat Ollama lokal.

[English](README.md) | [Cara pakai](docs/usage.id.md) | [Hasil](docs/results.md) | [Gambaran proyek](docs/project.md)

## Isinya apa?

- Notebook dari pengecekan respons model asli sampai ekstraksi aktivasi dan perubahan bobot.
- Konversi model ke Hugging Face dan GGUF F16.
- Setup Ollama lokal dan evaluasi respons.
- Metrik hasil run, grafik perbandingan, dan pengecekan ringan tanpa GPU.

Alurnya berakhir di evaluasi lewat Ollama lokal. Dataset prompt eksperimen disertakan di `data/`. Bobot model saat ini masih tersimpan lokal.

## Mulai dari sini

Kalau hanya ingin melihat hasilnya, buka tabel dan grafik di bawah. Untuk mengecek repository, gunakan Python 3.10 atau lebih baru:

```bash
python scripts/check_repo.py
python -m unittest discover -s tests -v
```

Untuk membuat ulang grafik tanpa memuat model:

```bash
python -m pip install -r requirements-report.txt
python scripts/plot_results.py
```

Untuk menjalankan alur model, ikuti [panduan penggunaan](docs/usage.id.md). Di sana ada setup environment, format input, urutan notebook, dan perintah Ollama.

## Hasil yang tersimpan

| Model / runtime | Prompt | Refusal rate | ASR heuristik |
| --- | ---: | ---: | ---: |
| Asli / TransformerLens | 80 | 75,00% | 25,00% |
| Abliterated / Hugging Face | 80 | 33,75% | 63,75% |
| Abliterated / Ollama | 80 | 35,00% | 65,00% |

![Perbandingan metrik respons](results/reference/metrics.png)

Refusal rate dihitung dari keyword. ASR heuristik menghitung respons non-refusal yang lebih panjang dari 50 karakter, bukan memeriksa apakah jawabannya benar-benar berbahaya. Set prompt antar-run juga berbeda dan ada overlap data ekstraksi/evaluasi. Jadi, baca angka ini sebagai hasil run yang tersimpan. Penjelasannya ada di [catatan hasil](docs/results.md).

## Alur dan folder

```text
Baseline -> Aktivasi -> Screening arah -> Perubahan bobot
         -> Hugging Face -> GGUF F16 -> Ollama -> Metrik respons
```

```text
notebooks/         Alur 00-08 dan alternatif Colab untuk tahap 02
src/               Konfigurasi, pembacaan data, generation, dan metrik
scripts/           Cek repository, buat grafik, dan kemas ZIP
tests/             Tes fungsi metrik
docs/              Panduan, gambaran proyek, hasil, dan troubleshooting
data/              CSV eksperimen; lihat data/README.md
models/            File model lokal; lihat models/README.md
results/reference/ Hasil tersimpan yang disertakan di repository
```

Run baru menulis file lokal di `results/` dan tidak menimpa hasil referensi secara otomatis. Token, bobot model, respons mentah, dan backup lokal tidak ikut Git maupun paket ZIP.

## Dokumentasi

- [Cara pakai](docs/usage.id.md): mulai dari setup sampai menjalankan notebook.
- [Usage guide](docs/usage.md): panduan dalam bahasa Inggris.
- [Gambaran proyek](docs/project.md): fungsi folder, modul, dan file output.
- [Troubleshooting](docs/troubleshooting.md): masalah setup dan runtime yang umum.
- [Catatan hasil](docs/results.md): arti metrik dan batas perbandingannya.
- [Setup GitHub](docs/github.md): nama repository, tag, dan perintah push.
- [Kredit](docs/attribution.md): tools dan referensi implementasi.

Repository ini berisi kode dan hasil run, bukan model siap unduh. Versi dependency dan data input bisa mengubah hasil. Alur model penuh tidak dijalankan ulang saat perapian ini.
