"""
app.py — Sentiment Analysis Dashboard
Evaluasi 48 Tahun UNS: Panggung Semar BEM UNS
Tema: Midnight Scholar — Dark Navy dengan aksen Amber & Cyan
Muhammad Rifai | Portfolio | Data Analytics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Sentiment UNS — Panggung Semar",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── TEMA: Midnight Scholar ───────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #070c18; color: #c8d8e8; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1225 0%, #070c18 100%);
    border-right: 1px solid rgba(245,200,66,0.15);
}
[data-testid="stSidebar"] * { color: #a8b8c8 !important; }

.hero-title {
    font-family: 'Inter', sans-serif;
    font-size: 2.2rem; font-weight: 700;
    background: linear-gradient(135deg, #f5c842 0%, #ffaa00 50%, #e8a44a 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; line-height: 1.15;
}
.hero-sub {
    font-size: 0.78rem; color: #4a6a7a; letter-spacing: 3px;
    text-transform: uppercase; margin-top: 6px;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(245,200,66,0.07) 0%, rgba(78,203,138,0.04) 100%);
    border: 1px solid rgba(245,200,66,0.2);
    border-top: 2px solid #f5c842;
    border-radius: 12px; padding: 18px 14px; text-align: center;
}
.kpi-number {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.9rem; font-weight: 600; color: #f5c842; line-height: 1;
}
.kpi-label { font-size: 0.65rem; color: #4a6a7a; text-transform: uppercase; letter-spacing: 2px; margin-top: 5px; }
.kpi-sub   { font-size: 0.75rem; color: #7a9ab0; margin-top: 3px; }

.section-title {
    font-size: 1rem; font-weight: 600; color: #f5c842;
    border-left: 3px solid #f5c842; padding-left: 10px; margin: 18px 0 12px;
}
.insight-box {
    background: rgba(245,200,66,0.04); border: 1px solid rgba(245,200,66,0.15);
    border-left: 3px solid #4ecb8a; border-radius: 8px;
    padding: 12px 16px; margin: 10px 0; font-size: 0.83rem;
    color: #a8c8d8; line-height: 1.65;
}
.quote-box {
    background: rgba(78,203,138,0.04); border: 1px solid rgba(78,203,138,0.2);
    border-left: 3px solid #4ecb8a; border-radius: 0 8px 8px 0;
    padding: 12px 16px; margin: 8px 0; font-size: 0.85rem;
    color: #a8c8d8; line-height: 1.65; font-style: italic;
}
.sent-pos { background: rgba(78,203,138,0.12); border:1px solid rgba(78,203,138,0.35);
            color:#4ecb8a; border-radius:20px; padding:3px 12px; font-size:0.78rem;
            font-weight:600; display:inline-block; }
.sent-neg { background: rgba(224,92,110,0.12); border:1px solid rgba(224,92,110,0.35);
            color:#e05c6e; border-radius:20px; padding:3px 12px; font-size:0.78rem;
            font-weight:600; display:inline-block; }
.sent-neu { background: rgba(245,200,66,0.12); border:1px solid rgba(245,200,66,0.35);
            color:#f5c842; border-radius:20px; padding:3px 12px; font-size:0.78rem;
            font-weight:600; display:inline-block; }
.source-tag {
    font-family: 'JetBrains Mono', monospace; font-size: 0.68rem;
    color: #4a8a9a; background: rgba(74,138,154,0.1);
    border: 1px solid rgba(74,138,154,0.25); border-radius: 4px;
    padding: 2px 8px; display: inline-block; margin-right: 4px;
}
hr { border-color: rgba(245,200,66,0.1) !important; }
.stTabs [data-baseweb="tab"] { font-family: 'Inter',sans-serif; color: #4a6a7a; }
.stTabs [aria-selected="true"] { color:#f5c842 !important; border-bottom:2px solid #f5c842 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Plotly Theme ─────────────────────────────────────────────────
PT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(7,12,24,0.8)",
    font=dict(family="Inter, sans-serif", color="#a8b8c8", size=11),
    title_font=dict(family="Inter, sans-serif", color="#f5c842", size=13),
)
AMBER = "#f5c842"; GREEN = "#4ecb8a"; RED = "#e05c6e"
TEAL  = "#4ecb8a"; BLUE  = "#4a9fd4"; PURPLE = "#9b8dc4"
SENT_COL = {"Positif": GREEN, "Negatif": RED, "Netral": AMBER}

# ─── Data & NLP Engine ────────────────────────────────────────────
RAW_TEXT = """
Kamis (21/3), Aliansi Badan Eksekutif Mahasiswa (BEM) Universitas Sebelas Maret menggelar
Panggung Semar bertajuk Evaluasi 48 Tahun UNS yang bertempat di Boulevard UNS.
Panggung ini diisi dengan pertunjukan bakat seperti monolog puisi, orasi, drama, hingga
penampilan band oleh perwakilan tiap fakultas. Selain penampilan bakat, Aliansi BEM UNS
juga membagikan takjil ramadan gratis kepada mahasiswa dan masyarakat sekitar.
Panggung Semar dimulai sejak pukul 15.30 hingga 18.00 sore yang dihadiri oleh puluhan mahasiswa.

Presiden BEM UNS, Agung Lucky mengatakan bahwa Panggung Semar ini dilaksanakan setiap tahun
sebagai bentuk mahasiswa mengekspresikan aspirasi untuk UNS. Agung juga menjelaskan terkait
perubahan lokasi yang sebelumnya bertempat di danau UNS tetapi mendadak pindah ke area Boulevard UNS.

Tidak diketahui sebelumnya, ijin juga sudah dibuat dan masuk ke satpam serta rektorat dan disetujui.
Ternyata hari-H itu tiba-tiba diketahui ada bentrok jadwal dengan acara dari kelompok lain di danau UNS.
Akhirnya kita berpindah tempat dan mengalah ke Boulevard UNS.

Panggung Semar ini diawali dengan pembacaan puisi oleh temen-teman Teater Tesa perwakilan dari
Fakultas Ilmu Budaya. Kemudian dilanjutkan dengan orasi dari perwakilan Fakultas Hukum,
serta penampilan drama oleh perwakilan Fakultas Ilmu Sosial dan Politik dan beberapa
penampilan bakat oleh fakultas lain.

Muhammad Rifai, mahasiswa Fakultas Matematika dan Ilmu Pengetahuan Alam jurusan
Statistika menjadi perwakilan pada Panggung Semar kali ini. Rifai bermonolog dengan puisi
ciptaannya berjudul Termenung.

Maknanya tentang mahasiswa yang termenung, bangga masuk UNS tapi dia susah untuk mengekspresikan
diri. Alasan keterbatasan seperti sarana prasarana dan lainnya buat dia mengekspresikan sebuah karya.

Muhammad Rifai juga memberikan harapan kepada UNS agar menjadi lebih baik dan lebih maju kedepannya.
Rifai juga berharap kepada petinggi kampus UNS supaya lebih terbuka dan menindaklanjuti
aspirasi-aspirasi dari mahasiswa.

Tanggapan juga diberikan oleh Lukman sebagai Wakil Presiden BEM UNS. Lukman mengatakan bahwa
panggung ini merupakan wujud kecintaan keluarga besar mahasiswa UNS sebagai hadiah untuk perayaan
Dies Natalis UNS ke-48. Dia menegaskan bentuk wujud kecintaan itu adalah mengingatkan pimpinan
kampus untuk lebih memperhatikan kebijakan yang berorientasi pada mutu pendidikan.

Hari ini atas nama BEM UNS, saya memberikan rapor merah kepada Pimpinan UNS karena
ketidakberhasilannya dalam menangani permasalahan yang ada di UNS seperti penghapusan SPI 0 rupiah
bagi kuota mandiri dan sarana prasarana yang kurang mendukung pembelajaran.
"""

STOPWORDS = {
    'yang','dan','di','ke','dari','ini','itu','ada','dengan','untuk','pada','dalam',
    'adalah','juga','oleh','atau','tidak','akan','telah','sudah','serta','sebagai',
    'dari','agar','atas','bagi','tapi','namun','tetapi','karena','saat','ketika',
    'pun','bahwa','jika','maka','hanya','kita','dia','mereka','kami','saya',
    'kamu','ia','nya','hingga','sejak','setelah','sebelum','setiap','lain',
    'beberapa','seperti','hal','cara','pihak','waktu','hari','tahun','tempat',
    'kali','jadi','menjadi','tentang','terkait','tutur','pungkas','kata','ujar',
    'mengatakan','menyatakan','menjelaskan','diketahui','bertempat','bertajuk',
    'merupakan','dilanjutkan','diawali','diisi','dimulai','dihadiri','sudah',
    'masuk','pindah','diberikan','uns','bem','mahasiswa','fakultas','boulevard',
    'danau','area','acara','perwakilan','semar','panggung','rifai','agung','lukman',
}

LEXICON_POS = {
    'bangga':2,'harapan':2,'berharap':2,'semangat':2,'kecintaan':2,
    'antusias':2,'kreativitas':2,'bakat':1,'kualitas':1,'baik':1,'maju':2,
    'terbuka':1,'apresiasi':2,'mendukung':1,'meriah':2,'gratis':1,'hadiah':2,
    'aspirasi':1,'ekspresi':1,'mengekspresikan':1,'penampilan':1,'pertunjukan':1,
    'memperhatikan':1,'menindaklanjuti':1,'perayaan':1,'wujud':1,'disetujui':1,
    'membagikan':1,'memberikan':1,'mewakili':1,'mengingatkan':1,'inovatif':2,
}
LEXICON_NEG = {
    'merah':3,'ketidakberhasilannya':3,'gagal':3,'permasalahan':2,
    'masalah':2,'kurang':2,'susah':2,'keterbatasan':2,'kekurangan':2,
    'bentrok':2,'mendadak':1,'mengalah':1,'penghapusan':2,'termenung':1,
    'evaluasi':1,'kritik':2,'kritis':1,'tidak':1,'belum':1,'rapor':2,
}
NEGATION    = {'tidak','bukan','belum','tanpa','jangan','kurang'}
INTENSIFIER = {'sangat':1.5,'sekali':1.3,'paling':1.4,'cukup':0.8,'agak':0.7}

def sentiment_score(text):
    tc = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    tokens = tc.split()
    pos, neg, pw, nw = 0, 0, [], []
    for i, t in enumerate(tokens):
        mult = INTENSIFIER.get(tokens[i-1], 1.0) if i > 0 else 1.0
        negated = any(tokens[max(0,i-j)] in NEGATION for j in range(1,3))
        if t in LEXICON_POS:
            s = LEXICON_POS[t] * mult
            if negated: neg+=s*0.5; nw.append(f'¬{t}')
            else: pos+=s; pw.append(t)
        elif t in LEXICON_NEG:
            s = LEXICON_NEG[t] * mult
            if negated: pos+=s*0.5; pw.append(f'¬{t}')
            else: neg+=s; nw.append(t)
    compound = round((pos-neg)/max(pos+neg,1), 4)
    label = 'Positif' if compound>0.15 else ('Negatif' if compound<-0.15 else 'Netral')
    return {'pos':round(pos,2),'neg':round(neg,2),'compound':compound,
            'label':label,'pos_words':pw,'neg_words':nw}

def preprocess(text):
    text = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
    return [t for t in re.findall(r'\b[a-z]{3,}\b', text) if t not in STOPWORDS]

def split_sentences(text):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip())>20]

# ─── Precompute All Data ───────────────────────────────────────────
@st.cache_data
def compute_all():
    paras_raw = [p.strip() for p in RAW_TEXT.strip().split('\n\n') if p.strip()]
    para_labels = [
        'P1 · Pembukaan — Deskripsi Acara',
        'P2 · Agung — Konteks Lokasi',
        'P3 · Kutipan Agung — Perubahan Lokasi',
        'P4 · Rundown Acara per Fakultas',
        'P5 · Profil Rifai — Perwakilan FMIPA',
        'P6 · Kutipan Rifai — Makna Puisi',
        'P7 · Harapan Rifai untuk UNS',
        'P8 · Lukman — Wujud Kecintaan',
        'P9 · Kutipan Lukman — Rapor Merah',
    ]
    rows = []
    for i, (txt, lbl) in enumerate(zip(paras_raw, para_labels[:len(paras_raw)])):
        r = sentiment_score(txt)
        rows.append({'id':i+1,'label':lbl,'text':txt,
                     'compound':r['compound'],'label_sent':r['label'],
                     'pos':r['pos'],'neg':r['neg'],
                     'pos_words':', '.join(r['pos_words'][:4]),
                     'neg_words':', '.join(r['neg_words'][:4]),
                     'word_count':len(txt.split())})
    df_para = pd.DataFrame(rows)

    # Sentences
    sents = []
    for _, row in df_para.iterrows():
        for s in split_sentences(row['text']):
            r = sentiment_score(s)
            sents.append({'para_id':row['id'],'sentence':s,
                          'compound':r['compound'],'label':r['label'],
                          'pos_words':', '.join(r['pos_words'][:3]),
                          'neg_words':', '.join(r['neg_words'][:3])})
    df_sent = pd.DataFrame(sents)

    # Narasumber
    quotes = {
        'Agung Lucky\n(Presiden BEM)': (
            'Panggung Semar ini dilaksanakan setiap tahun sebagai bentuk mahasiswa '
            'mengekspresikan aspirasi untuk UNS. Tidak diketahui sebelumnya, ijin juga '
            'sudah dibuat dan disetujui. Ternyata ada bentrok jadwal. Akhirnya mengalah.'
        ),
        'Muhammad Rifai\n(Mhsw FMIPA)': (
            'Maknanya tentang mahasiswa yang termenung, bangga masuk UNS tapi susah '
            'mengekspresikan diri karena keterbatasan sarana prasarana. Memberikan harapan '
            'kepada UNS agar menjadi lebih baik dan maju. Berharap lebih terbuka.'
        ),
        'Lukman\n(Wakil Presiden BEM)': (
            'Panggung ini merupakan wujud kecintaan mahasiswa sebagai hadiah perayaan '
            'Dies Natalis. Mengingatkan pimpinan kampus memperhatikan mutu pendidikan. '
            'Memberikan rapor merah karena ketidakberhasilannya menangani permasalahan '
            'penghapusan SPI dan sarana prasarana yang kurang mendukung.'
        ),
    }
    nar_rows = []
    for name, txt in quotes.items():
        r = sentiment_score(txt)
        nar_rows.append({'narasumber':name,'compound':r['compound'],
                         'label':r['label'],'pos':r['pos'],'neg':r['neg'],
                         'pos_words':', '.join(r['pos_words'][:4]),
                         'neg_words':', '.join(r['neg_words'][:4])})
    df_nar = pd.DataFrame(nar_rows)

    # ABSA
    aspects = {
        'Kegiatan & Event':      ['panggung','penampilan','bakat','puisi','orasi','drama','band',
                                   'monolog','pertunjukan','meriah','hadir','takjil','gratis','boulevard'],
        'Sarana & Prasarana':    ['sarana','prasarana','fasilitas','ruang','lokasi','danau',
                                   'tempat','infrastruktur','mendukung','kurang'],
        'Kebijakan Kampus':      ['kebijakan','spi','seleksi','mandiri','rektorat','pimpinan',
                                   'rapor','ketidakberhasilan','penghapusan','permasalahan'],
        'Mutu Pendidikan':       ['mutu','pendidikan','kualitas','tenaga','pendidik',
                                   'pembelajaran','akademik','jurusan','fakultas','proses'],
        'Aspirasi Mahasiswa':    ['aspirasi','harapan','ekspresi','mengekspresikan','berharap',
                                   'mengingatkan','kecintaan','wujud','bangga','suara'],
    }
    all_sents = split_sentences(RAW_TEXT)
    absa_rows = []
    for asp, kws in aspects.items():
        rel = [s for s in all_sents if any(k in s.lower() for k in kws)]
        r = sentiment_score(' '.join(rel)) if rel else {'compound':0,'label':'Netral','pos':0,'neg':0}
        absa_rows.append({'aspek':asp,'compound':r['compound'],'label':r['label'],
                          'kalimat_relevan':len(rel),'pos':r.get('pos',0),'neg':r.get('neg',0)})
    df_absa = pd.DataFrame(absa_rows)

    # Word freq
    tokens = preprocess(RAW_TEXT)
    freq = Counter(tokens)

    # Overall
    overall = sentiment_score(RAW_TEXT)

    return df_para, df_sent, df_nar, df_absa, freq, overall

df_para, df_sent, df_nar, df_absa, word_freq, overall = compute_all()

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:18px 0 10px;'>
        <div style='font-family:JetBrains Mono,monospace; font-size:1rem;
                    color:#f5c842; letter-spacing:2px;'>◉ SENTIMENT</div>
        <div style='font-size:0.6rem; color:#2a4a5a; letter-spacing:3px;
                    text-transform:uppercase; margin-top:4px;'>UNS Analytics</div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("**◉ Navigate**")
    page = st.radio("", [
        "Overview",
        "Analisis Paragraf",
        "Analisis Narasumber",
        "Aspect-Based (ABSA)",
        "Word Explorer",
        "Teks Asli",
    ], label_visibility="collapsed")

    st.divider()
    st.markdown(f"""
    <div style='font-size:0.65rem; color:#2a4a5a; text-align:center; line-height:2;'>
    📰 saluransebelas.com<br>
    25 Maret 2024<br>
    {len(df_sent)} kalimat · {len(df_para)} paragraf<br>
    {sum(word_freq.values())} kata (setelah filter)
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: OVERVIEW
# ══════════════════════════════════════════════════════════════════
if page == "Overview":
    st.markdown("""
    <div style='padding:20px 0 14px;'>
        <div class='hero-title'>ANALISIS SENTIMEN<br>BERITA UNS</div>
        <div class='hero-sub'>Evaluasi 48 Tahun UNS · Panggung Semar BEM UNS · 2024</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-bottom:12px;'>
        <span class='source-tag'>📰 saluransebelas.com</span>
        <span class='source-tag'>25 Maret 2024</span>
        <span class='source-tag'>Penulis: Veri Nugroho & Tiara Nur A.</span>
    </div>""", unsafe_allow_html=True)
    st.divider()

    # KPIs
    sent_label = overall['label']
    sent_color = SENT_COL[sent_label]
    c1,c2,c3,c4,c5 = st.columns(5)
    kpis = [
        (sent_label,           "Sentimen Dominan",     f"compound {overall['compound']:+.3f}"),
        (f"{overall['compound']:+.3f}", "Compound Score", "−1 negatif · +1 positif"),
        (str(len(df_sent)),    "Total Kalimat",        "dianalisis"),
        (str(len(df_para)),    "Paragraf",             "unit analisis"),
        (str(len(word_freq)),  "Unique Tokens",        "setelah stopword removal"),
    ]
    for col, (num, label, sub) in zip([c1,c2,c3,c4,c5], kpis):
        with col:
            st.markdown(f"""
            <div class='kpi-card'>
                <div class='kpi-number' style='font-size:1.5rem; color:{sent_color if label=="Sentimen Dominan" else AMBER};'>{num}</div>
                <div class='kpi-label'>{label}</div>
                <div class='kpi-sub'>{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns([1.5, 1])

    with col_a:
        st.markdown("<div class='section-title'>Alur Sentimen Sepanjang Berita</div>", unsafe_allow_html=True)
        fig = go.Figure()
        bar_cols = [SENT_COL[l] for l in df_para['label_sent']]
        fig.add_trace(go.Bar(
            x=df_para['id'], y=df_para['compound'],
            marker_color=bar_cols, opacity=0.8,
            text=[f"P{i}<br>{c:+.2f}" for i,c in zip(df_para['id'],df_para['compound'])],
            textposition='outside', textfont=dict(size=9),
            hovertemplate='<b>%{customdata}</b><br>Score: %{y:.3f}<extra></extra>',
            customdata=df_para['label']
        ))
        fig.add_hline(y=0, line_color='#2a3a4e', line_width=1.5)
        fig.add_hline(y=df_para['compound'].mean(), line_dash='dot', line_color=AMBER,
                      line_width=1.5,
                      annotation_text=f"Mean: {df_para['compound'].mean():+.3f}",
                      annotation_font_color=AMBER, annotation_font_size=9)
        fig.update_layout(**PT, height=300, margin=dict(l=0,r=0,t=20,b=0),
                          xaxis=dict(title='Paragraf', tickvals=list(df_para['id']),
                                     ticktext=[f'P{i}' for i in df_para['id']]),
                          yaxis=dict(title='Compound Score', range=[-1.1,1.1]),
                          showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Distribusi Sentimen Kalimat</div>", unsafe_allow_html=True)
        sent_dist = df_sent['label'].value_counts().reset_index()
        sent_dist.columns = ['label','count']
        fig2 = go.Figure(go.Pie(
            labels=sent_dist['label'], values=sent_dist['count'],
            hole=0.62,
            marker=dict(colors=[SENT_COL[l] for l in sent_dist['label']],
                        line=dict(color='#070c18', width=2)),
            textinfo='percent+label', textfont=dict(size=11)
        ))
        fig2.update_layout(**PT, height=300, margin=dict(l=0,r=0,t=20,b=0),
                           showlegend=False,
                           annotations=[dict(text=f"{sent_label}<br>{overall['compound']:+.2f}",
                                            x=0.5,y=0.5,
                                            font=dict(size=12,color=sent_color),
                                            showarrow=False)])
        st.plotly_chart(fig2, use_container_width=True)

    # Insight
    st.markdown(f"""
    <div class='insight-box'>
    ◉ <b>Ringkasan:</b> Berita ini menampilkan <b>dualitas sentimen</b> — semangat mahasiswa dalam
    kegiatan Panggung Semar (positif) berdampingan dengan kritik tajam terhadap kebijakan kampus
    yang diwujudkan dalam simbol "rapor merah" BEM UNS. Sentimen keseluruhan: <b>{sent_label}</b>
    (score: {overall['compound']:+.3f}).
    </div>""", unsafe_allow_html=True)

    # Kalimat ekstrem
    st.markdown("<div class='section-title'>Kalimat Paling Negatif & Paling Positif</div>", unsafe_allow_html=True)
    col_neg, col_pos = st.columns(2)
    with col_neg:
        st.markdown("🔴 **Paling Negatif**")
        for _, row in df_sent.nsmallest(3,'compound').iterrows():
            st.markdown(f"""<div class='quote-box' style='border-left-color:{RED};'>
            <span style='color:{RED}; font-size:0.7rem; font-weight:600;'>[{row['compound']:+.3f}]</span><br>
            {row['sentence'][:120]}{'...' if len(row['sentence'])>120 else ''}
            </div>""", unsafe_allow_html=True)
    with col_pos:
        st.markdown("🟢 **Paling Positif**")
        for _, row in df_sent.nlargest(3,'compound').iterrows():
            st.markdown(f"""<div class='quote-box'>
            <span style='color:{GREEN}; font-size:0.7rem; font-weight:600;'>[{row['compound']:+.3f}]</span><br>
            {row['sentence'][:120]}{'...' if len(row['sentence'])>120 else ''}
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: ANALISIS PARAGRAF
# ══════════════════════════════════════════════════════════════════
elif page == "Analisis Paragraf":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>ANALISIS PARAGRAF</div>", unsafe_allow_html=True)
    st.divider()

    # Heatmap compound per paragraf
    st.markdown("<div class='section-title'>Compound Score per Paragraf</div>", unsafe_allow_html=True)
    fig = go.Figure()
    bar_cols = [SENT_COL[l] for l in df_para['label_sent']]
    fig.add_trace(go.Bar(
        x=[f"P{i}" for i in df_para['id']],
        y=df_para['compound'],
        marker=dict(color=bar_cols, opacity=0.85,
                    line=dict(color='#070c18', width=0.5)),
        text=[f"{c:+.3f}" for c in df_para['compound']],
        textposition='outside', textfont=dict(size=10),
        hovertemplate='<b>%{customdata}</b><br>Score: %{y:.4f}<extra></extra>',
        customdata=df_para['label']
    ))
    fig.add_hline(y=0, line_color='#2a3a4e', line_width=1.5, line_dash='dash')
    fig.update_layout(**PT, height=320, margin=dict(l=0,r=0,t=20,b=0),
                      yaxis=dict(title='Compound Score', range=[-1.2,1.2]),
                      xaxis_title='Paragraf', showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Pos vs Neg stacked
    st.markdown("<div class='section-title'>Pos Score vs Neg Score</div>", unsafe_allow_html=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(name='Pos Score', x=[f"P{i}" for i in df_para['id']],
                          y=df_para['pos'], marker_color=GREEN, opacity=0.8))
    fig2.add_trace(go.Bar(name='Neg Score', x=[f"P{i}" for i in df_para['id']],
                          y=-df_para['neg'], marker_color=RED, opacity=0.8))
    fig2.add_hline(y=0, line_color='#2a3a4e', line_width=1)
    fig2.update_layout(**PT, height=280, margin=dict(l=0,r=0,t=10,b=0),
                       barmode='relative', yaxis_title='Score',
                       legend=dict(orientation='h', y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

    # Detail per paragraf
    st.markdown("<div class='section-title'>Detail Per Paragraf</div>", unsafe_allow_html=True)
    for _, row in df_para.iterrows():
        sent_class = 'sent-pos' if row['label_sent']=='Positif' else ('sent-neg' if row['label_sent']=='Negatif' else 'sent-neu')
        with st.expander(f"P{row['id']} — {row['label']}"):
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                st.markdown(f"**Score:** `{row['compound']:+.4f}`")
                st.markdown(f"**Label:** <span class='{sent_class}'>{row['label_sent']}</span>", unsafe_allow_html=True)
            with col2:
                st.markdown(f"**Pos Score:** `{row['pos']}`")
                st.markdown(f"**Neg Score:** `{row['neg']}`")
            with col3:
                if row['pos_words']:
                    st.markdown(f"🟢 Kata positif: `{row['pos_words']}`")
                if row['neg_words']:
                    st.markdown(f"🔴 Kata negatif: `{row['neg_words']}`")
            st.markdown(f"<div class='quote-box'>{row['text'][:400]}{'...' if len(row['text'])>400 else ''}</div>",
                        unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: ANALISIS NARASUMBER
# ══════════════════════════════════════════════════════════════════
elif page == "Analisis Narasumber":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>ANALISIS NARASUMBER</div>", unsafe_allow_html=True)
    st.divider()

    # KPI per narasumber
    cols = st.columns(3)
    narasumber_info = {
        'Agung Lucky\n(Presiden BEM)':    ('Presiden BEM UNS', 'AL'),
        'Muhammad Rifai\n(Mhsw FMIPA)':  ('Mahasiswa FMIPA Statistika', 'MR'),
        'Lukman\n(Wakil Presiden BEM)':   ('Wakil Presiden BEM UNS', 'LK'),
    }
    for col, (_, row) in zip(cols, df_nar.iterrows()):
        color = SENT_COL[row['label']]
        info  = narasumber_info.get(row['narasumber'], ('—','??'))
        inits = info[1]
        with col:
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.2); border:1px solid {color}33;
                        border-top:2px solid {color}; border-radius:12px;
                        padding:18px 14px; text-align:center;'>
                <div style='width:44px;height:44px;border-radius:50%;
                            background:{color}22; border:1.5px solid {color}55;
                            display:flex;align-items:center;justify-content:center;
                            margin:0 auto 10px; font-weight:600; font-size:13px;
                            color:{color};'>{inits}</div>
                <div style='font-size:0.78rem;font-weight:600;color:{color};
                            margin-bottom:4px;'>{row['narasumber'].replace(chr(10),' ')}</div>
                <div style='font-size:0.68rem;color:#4a6a7a;margin-bottom:10px;'>{info[0]}</div>
                <div style='font-family:JetBrains Mono,monospace;font-size:1.4rem;
                            font-weight:600;color:{color};'>{row['compound']:+.3f}</div>
                <div style='font-size:0.65rem;color:{color}99;margin-top:4px;'>{row['label']}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Horizontal bar comparison
    st.markdown("<div class='section-title'>Perbandingan Compound Score</div>", unsafe_allow_html=True)
    nar_names_short = ['Agung (Presiden)', 'Rifai (FMIPA)', 'Lukman (Wakil)']
    fig = go.Figure()
    bar_cols = [SENT_COL[l] for l in df_nar['label']]
    fig.add_trace(go.Bar(
        y=nar_names_short[::-1],
        x=df_nar['compound'][::-1].values,
        orientation='h',
        marker=dict(color=bar_cols[::-1], opacity=0.85),
        text=[f"{v:+.3f}" for v in df_nar['compound'][::-1].values],
        textposition='outside', textfont=dict(size=11, color='#c8d8e8')
    ))
    fig.add_vline(x=0, line_color='#2a3a4e', line_width=1.5)
    fig.update_layout(**PT, height=240, margin=dict(l=0,r=80,t=10,b=0),
                      xaxis=dict(title='Compound Score', range=[-1.1,1.1]),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Pos vs Neg per narasumber
    st.markdown("<div class='section-title'>Pos vs Neg Score per Narasumber</div>", unsafe_allow_html=True)
    fig2 = make_subplots(rows=1, cols=3,
                         subplot_titles=['Agung (Presiden)','Rifai (FMIPA)','Lukman (Wakil)'])
    for i, (_, row) in enumerate(df_nar.iterrows(), 1):
        color = SENT_COL[row['label']]
        fig2.add_trace(go.Bar(
            x=['Positif','Negatif'], y=[row['pos'], row['neg']],
            marker_color=[GREEN, RED], opacity=0.8,
            text=[f"{row['pos']:.1f}", f"{row['neg']:.1f}"],
            textposition='outside', textfont=dict(size=10),
            showlegend=False
        ), row=1, col=i)
    fig2.update_layout(**PT, height=280, margin=dict(l=0,r=0,t=30,b=0))
    st.plotly_chart(fig2, use_container_width=True)

    # Interpretasi
    st.markdown("<div class='section-title'>Interpretasi Narasumber</div>", unsafe_allow_html=True)
    interpretations = [
        ('Agung Lucky', 'Netral–Positif', AMBER,
         'Tone cenderung informatif dan netral. Menyampaikan fakta perubahan lokasi tanpa nada kritis berlebih. Penggunaan kata "disetujui" dan konteks aspirasi menambah nuansa positif.'),
        ('Muhammad Rifai', 'Netral', BLUE,
         'Menyampaikan perspektif campuran yang realistis — bangga masuk UNS namun terhambat keterbatasan. Harapan untuk UNS lebih maju menunjukkan optimisme konstruktif mahasiswa.'),
        ('Lukman', 'Negatif', RED,
         '"Rapor merah" dan "ketidakberhasilannya" adalah ekspresi kritis paling eksplisit dalam berita. Sebagai Wakil Presiden BEM, nada kritisnya mencerminkan posisi resmi organisasi mahasiswa terhadap kebijakan kampus.'),
    ]
    for name, label, color, interp in interpretations:
        st.markdown(f"""
        <div style='background:rgba(0,0,0,0.2); border:1px solid {color}22;
                    border-left:3px solid {color}; border-radius:0 10px 10px 0;
                    padding:14px 16px; margin-bottom:10px;'>
            <div style='color:{color}; font-weight:600; font-size:0.88rem; margin-bottom:5px;'>
                {name} — <span style='font-weight:400'>{label}</span>
            </div>
            <div style='color:#a8b8c8; font-size:0.82rem; line-height:1.6;'>{interp}</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: ABSA
# ══════════════════════════════════════════════════════════════════
elif page == "Aspect-Based (ABSA)":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>ASPECT-BASED SENTIMENT</div>", unsafe_allow_html=True)
    st.markdown("Sentimen dianalisis per aspek tematik dari berita.", unsafe_allow_html=True)
    st.divider()

    # ABSA bar chart
    st.markdown("<div class='section-title'>Compound Score per Aspek</div>", unsafe_allow_html=True)
    absa_cols = [SENT_COL[l] for l in df_absa['label']]
    fig = go.Figure(go.Bar(
        y=df_absa['aspek'][::-1],
        x=df_absa['compound'][::-1].values,
        orientation='h',
        marker=dict(color=absa_cols[::-1], opacity=0.85),
        text=[f"{v:+.3f}" for v in df_absa['compound'][::-1].values],
        textposition='outside', textfont=dict(size=11, color='#c8d8e8'),
        hovertemplate='<b>%{y}</b><br>Score: %{x:.4f}<extra></extra>'
    ))
    fig.add_vline(x=0, line_color='#2a3a4e', line_width=1.5)
    fig.update_layout(**PT, height=360, margin=dict(l=0,r=80,t=10,b=0),
                      xaxis=dict(title='Compound Score', range=[-1.1,1.1]),
                      showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Detail cards per aspek
    st.markdown("<div class='section-title'>Detail per Aspek</div>", unsafe_allow_html=True)
    aspek_icons = {
        'Kegiatan & Event':   '🎭',
        'Sarana & Prasarana': '🏛️',
        'Kebijakan Kampus':   '📋',
        'Mutu Pendidikan':    '📚',
        'Aspirasi Mahasiswa': '💬',
    }
    aspek_desc = {
        'Kegiatan & Event':   'Panggung Semar berjalan meriah dengan berbagai penampilan dari tiap fakultas. Pembagian takjil gratis menambah nuansa positif acara.',
        'Sarana & Prasarana': 'Keluhan eksplisit tentang kurangnya sarana prasarana menjadi salah satu poin dalam puisi Rifai dan tuntutan BEM.',
        'Kebijakan Kampus':   'Aspek paling kritis — "rapor merah" dan tuntutan penghapusan SPI 0 rupiah mencerminkan ketidakpuasan mahasiswa terhadap kebijakan.',
        'Mutu Pendidikan':    'Isu kualitas tenaga pendidik dan proses seleksi disampaikan sebagai bagian dari evaluasi Dies Natalis UNS ke-48.',
        'Aspirasi Mahasiswa': 'Harapan dan aspirasi mahasiswa disampaikan secara konstruktif — bangga pada UNS sekaligus mendorong perbaikan.',
    }
    cols = st.columns(2)
    for i, (_, row) in enumerate(df_absa.iterrows()):
        color = SENT_COL[row['label']]
        icon  = aspek_icons.get(row['aspek'], '●')
        desc  = aspek_desc.get(row['aspek'], '')
        with cols[i % 2]:
            st.markdown(f"""
            <div style='background:rgba(0,0,0,0.2); border:1px solid {color}22;
                        border-top:2px solid {color}; border-radius:10px;
                        padding:14px; margin-bottom:10px;'>
                <div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>
                    <span style='font-size:18px;'>{icon}</span>
                    <span style='color:{color}; font-weight:600; font-size:0.88rem;'>{row['aspek']}</span>
                    <span style='margin-left:auto; font-family:JetBrains Mono,monospace;
                                 font-size:0.85rem; color:{color};'>{row['compound']:+.3f}</span>
                </div>
                <div style='color:#6a8a9a; font-size:0.7rem; margin-bottom:6px;'>
                    {row['kalimat_relevan']} kalimat relevan · Pos: {row['pos']:.1f} · Neg: {row['neg']:.1f}
                </div>
                <div style='color:#a8b8c8; font-size:0.8rem; line-height:1.55;'>{desc}</div>
            </div>""", unsafe_allow_html=True)

    # Rangking visual
    st.markdown("<div class='section-title'>Ranking Aspek: Paling Positif → Paling Negatif</div>", unsafe_allow_html=True)
    df_absa_sorted = df_absa.sort_values('compound', ascending=False)
    for rank, (_, row) in enumerate(df_absa_sorted.iterrows(), 1):
        color = SENT_COL[row['label']]
        bar_w = abs(row['compound']) * 100
        bar_side = 'right' if row['compound'] >= 0 else 'left'
        st.markdown(f"""
        <div style='display:flex; align-items:center; gap:10px; margin-bottom:8px;'>
            <span style='color:{color}; font-size:13px; font-weight:600; min-width:18px;'>#{rank}</span>
            <span style='font-size:12px; color:#c8d8e8; min-width:180px;'>{row['aspek']}</span>
            <div style='flex:1; height:8px; background:#1a2a3e; border-radius:4px; overflow:hidden;'>
                <div style='width:{bar_w}%; height:100%; background:{color}; border-radius:4px;
                            float:{"left" if row["compound"]>=0 else "right"};'></div>
            </div>
            <span style='font-family:JetBrains Mono,monospace; font-size:12px;
                         color:{color}; min-width:55px; text-align:right;'>{row['compound']:+.3f}</span>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: WORD EXPLORER
# ══════════════════════════════════════════════════════════════════
elif page == "Word Explorer":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>WORD EXPLORER</div>", unsafe_allow_html=True)
    st.divider()

    col_a, col_b = st.columns([1,1])

    with col_a:
        st.markdown("<div class='section-title'>Top 20 Kata Paling Sering</div>", unsafe_allow_html=True)
        top20 = pd.DataFrame(word_freq.most_common(20), columns=['kata','frekuensi'])
        fig = go.Figure(go.Bar(
            x=top20['frekuensi'][::-1],
            y=top20['kata'][::-1],
            orientation='h',
            marker=dict(color=top20['frekuensi'][::-1],
                        colorscale=[[0,'#1a2a4e'],[0.5,'#4a9fd4'],[1,'#f5c842']],
                        showscale=False),
            text=top20['frekuensi'][::-1],
            textposition='outside', textfont=dict(size=9)
        ))
        fig.update_layout(**PT, height=500, margin=dict(l=0,r=40,t=10,b=0),
                          xaxis_title='Frekuensi')
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("<div class='section-title'>Kata Bermuatan Sentimen</div>", unsafe_allow_html=True)

        # Kumpulkan semua kata bermuatan
        all_pos_w, all_neg_w = [], []
        for p in df_para['text']:
            r = sentiment_score(p)
            all_pos_w.extend([w for w in r['pos_words'] if not w.startswith('¬')])
            all_neg_w.extend([w for w in r['neg_words'] if not w.startswith('¬')])

        pos_freq = Counter(all_pos_w)
        neg_freq = Counter(all_neg_w)

        # Combined diverging bar
        pos_top = pos_freq.most_common(8)
        neg_top = neg_freq.most_common(8)
        all_words = [w for w,_ in pos_top] + [w for w,_ in neg_top]
        all_vals  = [v for _,v in pos_top] + [-v for _,v in neg_top]
        all_cols  = [GREEN]*len(pos_top) + [RED]*len(neg_top)

        fig2 = go.Figure(go.Bar(
            x=all_vals, y=all_words,
            orientation='h',
            marker=dict(color=all_cols, opacity=0.85),
            text=[f"+{v}" if v>0 else str(v) for v in all_vals],
            textposition='outside', textfont=dict(size=9)
        ))
        fig2.add_vline(x=0, line_color='#2a3a4e', line_width=1.5)
        fig2.update_layout(**PT, height=500, margin=dict(l=0,r=40,t=10,b=0),
                           xaxis_title='← Negatif  |  Positif →',
                           showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter: word frequency vs sentiment weight
    st.markdown("<div class='section-title'>Bobot Sentimen vs Frekuensi</div>", unsafe_allow_html=True)
    scatter_data = []
    for w, freq in word_freq.items():
        if w in LEXICON_POS:
            scatter_data.append({'kata':w,'frekuensi':freq,'bobot':LEXICON_POS[w],'jenis':'Positif'})
        elif w in LEXICON_NEG:
            scatter_data.append({'kata':w,'frekuensi':freq,'bobot':-LEXICON_NEG[w],'jenis':'Negatif'})

    if scatter_data:
        df_sc = pd.DataFrame(scatter_data)
        fig3 = px.scatter(df_sc, x='frekuensi', y='bobot', color='jenis',
                          text='kata', size=[abs(b)*3+5 for b in df_sc['bobot']],
                          color_discrete_map={'Positif':GREEN, 'Negatif':RED})
        fig3.update_traces(textposition='top center', textfont=dict(size=8))
        fig3.add_hline(y=0, line_color='#2a3a4e', line_dash='dash')
        fig3.update_layout(**PT, height=360, margin=dict(l=0,r=0,t=10,b=0),
                           xaxis_title='Frekuensi dalam Teks',
                           yaxis_title='Bobot Sentimen (+ Positif / − Negatif)',
                           legend_title='Jenis')
        st.plotly_chart(fig3, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# PAGE: TEKS ASLI
# ══════════════════════════════════════════════════════════════════
elif page == "Teks Asli":
    st.markdown("<div class='hero-title' style='font-size:1.8rem;'>TEKS ASLI BERITA</div>", unsafe_allow_html=True)
    st.divider()

    st.markdown("""
    <div style='margin-bottom:14px;'>
        <span class='source-tag'>📰 saluransebelas.com</span>
        <span class='source-tag'>25 Maret 2024</span>
        <span class='source-tag'>Penulis: Veri Nugroho & Tiara Nur A.</span>
    </div>""", unsafe_allow_html=True)

    # Tampilkan per paragraf dengan highlight sentimen
    for _, row in df_para.iterrows():
        color = SENT_COL[row['label_sent']]
        sent_class = 'sent-pos' if row['label_sent']=='Positif' else ('sent-neg' if row['label_sent']=='Negatif' else 'sent-neu')
        st.markdown(f"""
        <div style='border-left:3px solid {color}; border-radius:0 8px 8px 0;
                    padding:12px 16px; margin-bottom:12px;
                    background:rgba(0,0,0,0.15);'>
            <div style='display:flex; align-items:center; gap:8px; margin-bottom:8px;'>
                <span style='font-size:0.68rem; color:#4a6a7a; font-family:JetBrains Mono,monospace;'>P{row['id']}</span>
                <span class='{sent_class}'>{row['label_sent']}</span>
                <span style='font-family:JetBrains Mono,monospace; font-size:0.72rem; color:{color};'>{row['compound']:+.3f}</span>
            </div>
            <div style='color:#c8d8e8; font-size:0.88rem; line-height:1.7;'>{row['text']}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class='insight-box'>
    🔗 <b>Sumber asli:</b>
    <a href='https://saluransebelas.com/evaluasi-48-tahun-uns-aliansi-bem-uns-gelar-panggung-semar-dan-bagi-takjil/'
       target='_blank' style='color:#4ecb8a;'>
    saluransebelas.com — Evaluasi 48 Tahun UNS: Aliansi BEM UNS Gelar Panggung SEMAR dan Bagi Takjil
    </a>
    </div>""", unsafe_allow_html=True)

# ─── Footer ───────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style='text-align:center; padding:8px 0;
     font-family:JetBrains Mono,monospace; font-size:0.62rem;
     color:#1a3a4a; letter-spacing:2px;'>
◉ SENTIMENT ANALYSIS PANGGUNG SEMAR UNS ◉ MUHAMMAD RIFAI ◉ ANALYTIFAI ◉
</div>""", unsafe_allow_html=True)
