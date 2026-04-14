# Analisis Sentimen Berita — Evaluasi 48 Tahun UNS

> **"Bagaimana sentimen mahasiswa UNS terhadap kebijakan kampus yang tercermin dalam pemberitaan Panggung Semar?"**

![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-Lexicon--Based-4ecb8a)
![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-F37626?logo=jupyter&logoColor=white)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b?logo=streamlit&logoColor=white)](https://sentimen-panggung-semar.streamlit.app/)
---

## Sumber Berita

| Atribut | Detail |
|---|---|
| **Judul** | Evaluasi 48 Tahun UNS: Aliansi BEM UNS Gelar Panggung SEMAR dan Bagi Takjil |
| **Tanggal** | 25 Maret 2024 |
| **Sumber** | [saluransebelas.com](https://saluransebelas.com/evaluasi-48-tahun-uns-aliansi-bem-uns-gelar-panggung-semar-dan-bagi-takjil/) |
| **Penulis** | Veri Nugroho dan Tiara Nur A. |
| **Topik** | Aspirasi mahasiswa, evaluasi kebijakan kampus, kegiatan seni BEM UNS |

---

## Problem

Berita ini meliput kegiatan Panggung Semar yang digelar Aliansi BEM UNS sebagai bentuk ekspresi aspirasi mahasiswa pada Dies Natalis UNS ke-48. Analisis ini menjawab:

1. Apakah berita ini secara keseluruhan bernada positif, negatif, atau netral?
2. Bagian mana yang paling kritis terhadap kampus?
3. Bagaimana nada masing-masing narasumber?
4. Aspek apa (kebijakan, sarana, kegiatan) yang paling bermuatan sentimen negatif?

---

## Metodologi

```
1. Text Preprocessing
   └── Tokenisasi regex · Lowercase · Custom stopword removal (Bahasa Indonesia)

2. Lexicon-Based Sentiment Analysis
   ├── Kamus 30+ kata positif + 25+ kata negatif (domain kampus & berita)
   ├── Bobot kata (1–3) berdasarkan intensitas
   ├── Negation handling (2-word window)
   └── Intensifier multiplier (sangat, sekali, paling)

3. Granular Analysis
   ├── Sentimen keseluruhan artikel
   ├── Sentimen per paragraf (9 paragraf)
   ├── Sentimen per kalimat
   └── Sentimen per narasumber (Agung, Rifai, Lukman)

4. Aspect-Based Sentiment Analysis (ABSA)
   └── 5 aspek: Kegiatan & Event · Sarana & Prasarana · Kebijakan Kampus
                Mutu Pendidikan · Aspirasi Mahasiswa

5. Visualisasi
   └── Word frequency · Sentiment flow · Narrator chart · ABSA · Dashboard
```

---

## Key Results

### Sentimen Keseluruhan
| Metrik | Nilai |
|---|---|
| Label | **Netral–Negatif** |
| Compound Score | -0.26 |
| Kata Positif Dominan | bangga, harapan, kecintaan, hadiah, aspirasi |
| Kata Negatif Dominan | rapor merah, ketidakberhasilannya, permasalahan, kurang |

### Sentimen per Narasumber
| Narasumber | Compound | Label |
|---|---|---|
| Agung Lucky (Presiden BEM) | +0.15 | Netral–Positif |
| Muhammad Rifai (Mhsw FMIPA) | -0.05 | Netral |
| Lukman (Wakil Presiden BEM) | -0.42 | **Negatif** |

### Aspect-Based Sentiment (ABSA)
| Aspek | Compound | Label |
|---|---|---|
| Kegiatan & Event | +0.38 | **Positif** |
| Aspirasi Mahasiswa | +0.18 | Positif |
| Mutu Pendidikan | -0.10 | Netral |
| Sarana & Prasarana | -0.31 | Negatif |
| Kebijakan Kampus | -0.58 | **Negatif** |

---

## Key Findings

> Berita ini menggambarkan **dualitas**: semangat mahasiswa dalam Panggung Semar (positif) berdampingan dengan kritik tajam terhadap kebijakan kampus, simbolisasi "rapor merah" dari BEM (negatif).

| Finding | Interpretasi |
|---|---|
| Kegiatan mendapat sentimen positif | Antusiasme mahasiswa tinggi — event berhasil dari sisi partisipasi |
| Kebijakan kampus paling negatif | Isu SPI & sarana-prasarana adalah pain point utama mahasiswa |
| Lukman paling kritis | Bahasa "rapor merah" dan "ketidakberhasilan" sangat bermuatan negatif |
| Rifai menyampaikan pesan campuran | Bangga pada UNS tapi terhambat keterbatasan — optimisme konstruktif |

---

## Visualizations

- `word_analysis.png` — Top 15 kata + kata bermuatan positif/negatif
- `sentiment_flow.png` — Alur compound score per paragraf + pos vs neg stacked bar
- `narrator_absa.png` — Sentimen per narasumber + ABSA per aspek
- `sentiment_dashboard.png` — Executive summary dashboard (4 panel)

---

## Cara Menjalankan

```bash
git clone https://github.com/ahmadripai047/sentiment-uns.git
cd sentiment-uns
pip install -r requirements.txt
jupyter notebook sentiment_uns.ipynb
```

**Tidak membutuhkan dataset eksternal** — teks artikel sudah embedded di notebook.

---

## Struktur Repo

```
sentiment-uns/
├── sentiment_uns.ipynb    ← Notebook utama (20 cells)
├── requirements.txt
├── README.md
└── images/
    ├── word_analysis.png
    ├── sentiment_flow.png
    ├── narrator_absa.png
    └── sentiment_dashboard.png
```

---

## 👤 Author

**Muhammad Rifai, S.Stat** *(yang juga disebutkan dalam berita sebagai perwakilan FMIPA)*
- GitHub: [@ahmadripai047](https://github.com/ahmadripai047)
- LinkedIn: [in/muhammad-rifai047](https://linkedin.com/in/muhammad-rifai047)

---

*Portfolio | Analytifai | Sumber: saluransebelas.com — Pers Mahasiswa UNS*
