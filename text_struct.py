import re
from collections import Counter

MIN_NODE_WORDS = 20   # ambang kata minimum agar suatu bagian layak diringkas

# ── Text helpers ──────────────────────────────────────────────────────────────
def extract_pdf(content: bytes) -> str:
    try:
        import fitz
        doc = fitz.open(stream=content, filetype="pdf")
        return "\x0c".join(page.get_text() for page in doc)
    except ImportError:
        return ""

PAGE_NUM_RE = re.compile(r'^\s*\d{1,4}\s*$')

def _norm_header(line: str) -> str:
    l = re.sub(r'\d+', '', line)
    return re.sub(r'\s+', ' ', l).strip().lower()

def _is_heading_like(line: str) -> bool:
    return bool(BAB_RE.match(line) or ALPHA_HEAD_RE.match(line) or NUM_HEAD_RE.match(line))

KDT_LINE_RE = re.compile(r'^[^\n]*\bhlm\b[^\n]*\bcm\.?[ \t]*$', re.MULTILINE | re.IGNORECASE)

def strip_colophon(text: str) -> str:
    """Hapus halaman hak cipta/identitas penerbit (judul, ISBN, Hak Cipta,
    Disclaimer, dll.) di awal dokumen - dikenali dari baris Katalog Dalam
    Terbitan (KDT) khas Kemendikbud (mis. 'x, 142 hlm. : 17,6 x 25 cm.').
    Jika tidak ditemukan, teks dibiarkan apa adanya."""
    m = KDT_LINE_RE.search(text[:5000])
    return text[m.end():] if m else text

def strip_running_headers(text: str) -> str:
    """Hapus header/footer yg berulang di pinggir setiap halaman (mis. judul
    buku yg dicetak di setiap halaman), berdasarkan posisi (baris pertama/
    terakhir tiap halaman), bukan jumlah kata - sehingga header pendek
    (1-3 kata) pun terdeteksi."""
    pages = text.split('\x0c')
    if len(pages) < 4:
        return text.replace('\x0c', '\n')

    edge_counts = Counter()
    for p in pages:
        plines = [l for l in p.split('\n') if l.strip()]
        if not plines:
            continue
        for edge in (plines[0], plines[-1]):
            if not PAGE_NUM_RE.match(edge.strip()) and not _is_heading_like(edge):
                key = _norm_header(edge)
                if key:
                    edge_counts[key] += 1

    n = len(pages)
    noise = {k for k, c in edge_counts.items() if c >= max(3, n * 0.3)}
    joined = text.replace('\x0c', '\n')
    if not noise:
        return joined

    out = [l for l in joined.split('\n') if _is_heading_like(l) or _norm_header(l) not in noise]
    return '\n'.join(out)

def strip_boilerplate(text: str) -> str:
    """Hapus baris nomor halaman, dan baris footer/header yg berulang
    sangat sering (>=5x) & tidak menyerupai heading bab/sub-bab/poin."""
    lines = text.split('\n')

    def norm(line):
        return re.sub(r'\s*\d+\s*$', '', line.strip())

    candidates = [l for l in lines if l.strip() and not PAGE_NUM_RE.match(l) and not _is_heading_like(l)]
    counts = Counter(norm(l) for l in candidates)
    noise  = {k for k, c in counts.items() if c >= 5 and 4 <= len(k.split()) <= 14}

    total_words   = sum(len(l.split()) for l in lines)
    removed_words = sum(len(l.split()) for l in candidates if norm(l) in noise)
    if total_words and removed_words / total_words > 0.25:
        noise = set()  # terlalu banyak yg cocok -> kemungkinan salah deteksi, batalkan

    out = []
    for l in lines:
        if PAGE_NUM_RE.match(l):
            continue
        if not _is_heading_like(l) and norm(l) in noise:
            continue
        out.append(l)
    return '\n'.join(out)

# Caption satu baris penuh: "Gambar 3.1. Sumpah Pemuda ..." (barisnya sendiri)
GAMBAR_LINE_RE = re.compile(r'^[ \t]*(?:Gambar|Tabel)\s+\d+\.\d+\b[^\n]*$', re.MULTILINE | re.IGNORECASE)
# Referensi "Gambar 3.1" / "Tabel 3.1." yg NYELIP di tengah paragraf (incl. titiknya)
GAMBAR_INLINE_RE = re.compile(r'(?:Gambar|Tabel)\s+\d+(?:\.\d+)*\.?', re.IGNORECASE)

def strip_figure_captions(text: str) -> str:
    """Hapus caption/ref gambar-tabel. Dua kasus:
    (1) caption satu baris penuh   -> buang seluruh barisnya;
    (2) ref 'Gambar 3.1' yg nyelip di tengah kalimat akibat tata letak PDF
        -> buang token-nya saja, lalu rapikan tanda baca yatim."""
    text = GAMBAR_LINE_RE.sub('', text)         # kasus 1
    text = GAMBAR_INLINE_RE.sub(' ', text)      # kasus 2
    text = re.sub(r'\s+([.,;:])', r'\1', text)  # " ." -> "."  (sisa hapus token)
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text

def strip_toc_lines(text: str) -> str:
    """Hapus baris daftar isi/daftar gambar (mis. 'Bab III ... ..... 43')
    yg dikenali dari deretan titik (dot-leader) sebelum nomor halaman."""
    return '\n'.join(l for l in text.split('\n') if not re.search(r'\.{4,}', l))

PETA_KONSEP_RE = re.compile(r'^[ \t]*Peta\s+Konsep\s*$', re.MULTILINE | re.IGNORECASE)

def strip_concept_maps(text: str) -> str:
    """Hapus blok 'Peta Konsep' (label diagram tanpa kalimat utuh) di awal
    bab - baris demi baris hingga baris pertama yg benar2 berupa kalimat
    (diakhiri . ! ?) atau cukup panjang (>8 kata), agar tidak nyambung jadi
    satu 'kalimat' raksasa dgn kalimat pembuka bab."""
    lines = text.split('\n')
    out, i = [], 0
    while i < len(lines):
        if PETA_KONSEP_RE.match(lines[i]):
            i += 1
            skipped = 0
            while i < len(lines) and skipped < 25:
                l = lines[i].strip()
                if l and (re.search(r'[.!?]\s*$', l) or len(l.split()) > 8):
                    break
                i += 1
                skipped += 1
            continue
        out.append(lines[i])
        i += 1
    return '\n'.join(out)

STAMP_LINE_RE = re.compile(
    r'^[ \t]*(?:ISBN:?\s*[\d\- ]+(?:\([^)]*\))?|REPUBLIK\s+INDONESIA,?\s*\d{4})[ \t]*$',
    re.MULTILINE | re.IGNORECASE,
)

def strip_stamp_lines(text: str) -> str:
    """Hapus baris 'stempel' identitas buku (ISBN, REPUBLIK INDONESIA <tahun>)
    yg berulang di awal tiap bab tapi lolos dari strip_boilerplate karena
    terlalu pendek (<4 kata)."""
    return '\n'.join(l for l in text.split('\n') if not STAMP_LINE_RE.match(l))

# Link & sitasi sumber yg sering nyelip di buku.
MD_LINK_RE   = re.compile(r'\[[^\]]*\]\([^)]*\)')                 # [teks](url) (kalau ada)
URL_RE       = re.compile(r'(?:https?://|www\.)\S+', re.IGNORECASE)
SUMBER_YEAR  = re.compile(r'\bSumber\s*:.*?\(\s*\d{4}\s*\)', re.IGNORECASE)  # "Sumber: ... (2012)"
SUMBER_TOKEN = re.compile(r'\bSumber\s*:\s*\S+', re.IGNORECASE)             # "Sumber: <url/teks>"

def strip_citations(text: str) -> str:
    """Hapus URL & sitasi 'Sumber: ... (tahun)' yg ikut ketarik dari buku."""
    text = MD_LINK_RE.sub(' ', text)
    text = SUMBER_YEAR.sub(' ', text)     # buang "Sumber: ... (2012)" sebagai satu kesatuan
    text = SUMBER_TOKEN.sub(' ', text)    # buang "Sumber: <token>" yg tanpa tahun
    text = URL_RE.sub(' ', text)          # buang URL yg masih nyangkut di mana pun
    text = re.sub(r'\[\s*\]|\(\s*\)', ' ', text)   # kurung kosong sisa hapus
    return re.sub(r'[ \t]{2,}', ' ', text)

# Label kotak aktivitas/pedagogis Kemdikbud yg bukan kalimat materi.
# Tambahkan sesuai buku kamu kalau ada yg lain.
ACTIVITY_RE = re.compile(
    r'\b(?:Siswa\s+Aktif|Tahukah\s+Kamu|Ayo[, ]+\w+|Mari\s+\w+|'
    r'Aktivitas(?:\s+(?:Siswa|Belajar))?|Lembar\s+Aktivitas|'
    r'Tugas(?:\s+(?:Mandiri|Kelompok|Individu))?|Refleksi|Rangkuman|'
    r'Uji\s+Kompetensi|Asesmen|Penilaian\s+Diri)\b',
    re.IGNORECASE,
)

def strip_activity_labels(text: str) -> str:
    """Hapus label kotak aktivitas (mis. 'Siswa Aktif') yg nyelip ke teks."""
    text = ACTIVITY_RE.sub(' ', text)
    return re.sub(r'[ \t]{2,}', ' ', text)

def clean_text(raw: str) -> str:
    raw = raw.replace('\x00', '').encode('utf-8', 'ignore').decode('utf-8')
    raw = re.sub(r'\xad[ \t]*', '', raw)  # soft hyphen (U+00AD) - sisa pemenggalan suku kata di PDF
    raw = re.sub(r'[-�]', ' ', raw)  # glyph bullet/ikon font & karakter rusak
    raw = strip_colophon(raw)
    raw = strip_running_headers(raw)
    raw = strip_toc_lines(raw)
    raw = strip_citations(raw)
    raw = strip_figure_captions(raw)
    raw = strip_activity_labels(raw)
    raw = strip_concept_maps(raw)
    raw = strip_stamp_lines(raw)
    raw = strip_boilerplate(raw)
    return re.sub(r'([A-Za-z])-\s+([a-z])', r'\1\2', raw)

# ── Pemecahan kalimat (DIPERBAIKI) ────────────────────────────────────────────
# Singkatan umum yg titiknya BUKAN akhir kalimat.
_ABBREV = ["dll", "dsb", "dst", "hlm", "no", "yg", "tsb", "drs", "ir",
           "prof", "dr", "ttd", "an", "sd"]
# Pisah kalimat HANYA bila setelah . ! ? diikuti spasi lalu huruf kapital/angka/kutip.
SENT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')

def split_sentences(text: str):
    text = re.sub(r'\s+', ' ', str(text)).strip()
    if not text:
        return []
    # 1) lindungi titik pada angka desimal: '3.1' -> '3<D>1'
    text = re.sub(r'(?<=\d)\.(?=\d)', '<D>', text)
    # 2) lindungi titik singkatan umum: 'dll.' -> 'dll<D>'
    for ab in _ABBREV:
        text = re.sub(rf'\b({re.escape(ab)})\.', r'\1<D>', text, flags=re.IGNORECASE)
    # 3) lindungi penanda daftar huruf tunggal di tengah teks: ' a. ' ' b. '
    text = re.sub(r'(?<=[ ,;(])([a-z])\.(?=\s)', r'\1<D>', text)
    # pecah kalimat
    out = []
    for s in SENT_RE.split(text):
        s = s.replace('<D>', '.').strip()
        if len(s) > 20:
            out.append(s)
    return out

# ── Deteksi struktur bab ──────────────────────────────────────────────────────
BAB_RE = re.compile(r'^\s*(?:BAB|Bab|CHAPTER)\s+(?P<mark>[IVXLCDM]+|\d+)\b(?P<title>[^\n]*)$', re.MULTILINE)
ALPHA_HEAD_RE = re.compile(r'^[ \t]*(?P<mark>[A-Z]\.)[ \t]+(?P<title>\S[^\n]*)$', re.MULTILINE)
NUM_HEAD_RE   = re.compile(r'^[ \t]*(?P<mark>\d+\.)[ \t]+(?P<title>\S[^\n]*)$', re.MULTILINE)
ALPHA_SEQUENCE = [f"{chr(c)}." for c in range(ord('A'), ord('Z') + 1)]
NUM_SEQUENCE   = [f"{i}." for i in range(1, 61)]
ROMAN_CHAPTER_SEQUENCE  = ["I","II","III","IV","V","VI","VII","VIII","IX","X",
                           "XI","XII","XIII","XIV","XV","XVI","XVII","XVIII","XIX","XX"]
ARABIC_CHAPTER_SEQUENCE = [str(i) for i in range(1, 21)]
BACKMATTER_RE = re.compile(
    r'^\s*(?:Glosarium|Daftar\s+Pustaka|Daftar\s+Sumber\s+Gambar|Indeks|'
    r'Profil\s+(?:Penulis|Penelaah|Editor|Ilustrator|Desainer))\b',
    re.MULTILINE | re.IGNORECASE,
)

FUNCTION_WORDS = {
    "di","ke","dari","dan","atau","yang","untuk","dengan","pada","dalam","atas","oleh",
    "sebagai","serta","akan","ini","itu","para","secara","agar","bagi","tanpa","antara",
    "melalui","terhadap","kepada","adalah","ialah","yaitu","si",
}
TRAILING_STOP = {
    "oleh","dan","atau","yang","untuk","dengan","pada","dari","ke","di","dalam","sebagai",
    "serta","adalah","ialah","atas","bagi","kepada","terhadap","antara","akan","agar",
    "tentang","ini","itu","para","secara","sebuah","suatu","maka","karena","jika",
    "apabila","menjadi","yaitu","dapat","telah","sudah","harus","tidak","bukan",
}

def is_toc_like(text):
    words = len(text.split())
    if words == 0: return True
    leaders = len(re.findall(r'\.{4,}', text))
    return (words < 500 and leaders >= 4) or (leaders / words > 0.05)

def is_heading_title(title):
    t = title.strip()
    if not t or t[-1] in '.,;:-–—': return False
    if re.search(r'\.{3,}', t): return False
    words = t.split()
    if not (1 <= len(words) <= 12): return False
    if words[-1].lower().strip('.,;:()"\'') in TRAILING_STOP: return False
    content = [w for w in words if w.lower().strip('()"\'') not in FUNCTION_WORDS]
    if not content: return False
    return sum(1 for w in content if w[:1].isupper()) / len(content) >= 0.6

def _nid(counter):
    counter[0] += 1
    return f"n{counter[0]}"

def split_by_headings(text, pattern, sequence, title_map=None, allow_empty_title=False):
    """Pecah teks pada baris yang cocok dgn pattern, lolos is_heading_title,
    DAN mengikuti urutan mark yg diharapkan (A,B,C,... atau 1,2,3,...).
    Mark yg keluar urutan (misal nama orang "R. Abdoelrahim") diabaikan
    agar tidak dianggap heading palsu. Elemen pertama selalu konten
    sebelum heading pertama (title="").

    Jika allow_empty_title=True, heading dgn judul kosong (judulnya ada di
    baris/halaman lain, mis. "BAB I" saja) juga diterima selama mark-nya
    sesuai urutan; title_map (mark -> judul lengkap dari Daftar Isi) dipakai
    utk mengisi judul heading semacam itu."""
    seq = iter(sequence)
    expected = next(seq, None)
    matches = []
    for m in pattern.finditer(text):
        if expected is None:
            break
        if m.group('mark') != expected:
            continue
        title = m.group('title')
        if is_heading_title(title) or (allow_empty_title and not title.strip()):
            matches.append(m)
            expected = next(seq, None)
    if not matches:
        return [{"title": "", "content": text}]
    segments = [{"title": "", "content": text[:matches[0].start()].strip()}]
    for i, m in enumerate(matches):
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(text)
        heading = re.sub(r'\s+', ' ', m.group(0)).strip()
        if title_map is not None and not m.group('title').strip():
            heading = title_map.get(m.group('mark'), heading)
        segments.append({"title": heading, "content": text[start:end].strip()})
    return segments

def _find_toc_run(text, sequence, max_gap_words=40):
    """Cari deretan heading 'BAB <angka>' berurutan (I,II,III,... atau
    1,2,3,...) yang berjarak sangat dekat satu sama lain - ciri Daftar Isi
    yang tidak memakai dot-leader (mis. semua judul bab tercantum baris demi
    baris di awal dokumen)."""
    run, idx = [], 0
    for m in BAB_RE.finditer(text):
        if idx >= len(sequence):
            break
        if m.group('mark') == sequence[idx]:
            if run and len(text[run[-1].end():m.start()].split()) >= max_gap_words:
                break
            run.append(m)
            idx += 1
        elif run:
            break
    return run if len(run) >= 3 else []

def split_chapters(text):
    """Pecah teks pada heading 'BAB <angka>' urut I,II,III,... (atau
    1,2,3,...). Jika ditemukan blok Daftar Isi tanpa dot-leader (lih.
    _find_toc_run), blok itu dibuang dari teks dan judulnya dipakai utk
    heading bab asli yang judulnya kosong (judul menyusul di halaman lain)."""
    for seq in (ROMAN_CHAPTER_SEQUENCE, ARABIC_CHAPTER_SEQUENCE):
        toc_run = _find_toc_run(text, seq)
        if not toc_run:
            continue
        title_map = {m.group('mark'): re.sub(r'\s+', ' ', m.group(0)).strip() for m in toc_run}
        text2 = text[:toc_run[0].start()] + text[toc_run[-1].end():]
        segs = split_by_headings(text2, BAB_RE, seq, title_map=title_map, allow_empty_title=True)
        if len(segs) > 1:
            return segs

    seg_roman  = split_by_headings(text, BAB_RE, ROMAN_CHAPTER_SEQUENCE)
    seg_arabic = split_by_headings(text, BAB_RE, ARABIC_CHAPTER_SEQUENCE)
    return seg_roman if len(seg_roman) >= len(seg_arabic) else seg_arabic

def build_point(title, content, counter, nodes):
    pid = _nid(counter)
    nodes[pid] = content
    return {"id": pid, "title": title, "word_count": len(content.split())}

def build_subchapter(title, content, counter, nodes):
    sid = _nid(counter)
    segments = split_by_headings(content, NUM_HEAD_RE, NUM_SEQUENCE)
    intro, point_segs = segments[0]["content"], segments[1:]
    nodes[sid] = intro if point_segs else content
    points = [
        build_point(seg["title"], seg["content"], counter, nodes)
        for seg in point_segs if len(seg["content"].split()) >= MIN_NODE_WORDS
    ]
    return {"id": sid, "title": title, "word_count": len(nodes[sid].split()), "points": points}

def build_chapter(title, content, counter, nodes):
    cid = _nid(counter)
    segments = split_by_headings(content, ALPHA_HEAD_RE, ALPHA_SEQUENCE)
    intro, sub_segs = segments[0]["content"], segments[1:]

    has_intro   = len(intro.split()) >= MIN_NODE_WORDS or not sub_segs
    intro_words = 0
    if has_intro:
        nodes[cid] = intro if sub_segs else content
        intro_words = len(nodes[cid].split())

    sub_chapters = [build_subchapter(seg["title"], seg["content"], counter, nodes) for seg in sub_segs]

    return {
        "id": cid,
        "title": title,
        "word_count": len(content.split()),
        "has_intro": has_intro,
        "intro_words": intro_words,
        "sub_chapters": sub_chapters,
    }

def build_structure(text):
    text = clean_text(text)
    counter = [0]
    nodes = {}

    segments = split_chapters(text)

    def norm_title(t):
        t = re.sub(r'\.{2,}', ' ', t)
        t = re.sub(r'\s+\d+\s*$', '', t)
        return re.sub(r'\s+', ' ', t).strip()

    def strip_bab_lines(t):
        return '\n'.join(l for l in t.split('\n') if not BAB_RE.match(l))

    raw = []
    pre = strip_bab_lines(segments[0]["content"])
    if len(pre.split()) > 30 and not is_toc_like(pre):
        raw.append({"title": "Pendahuluan", "content": pre})

    chapter_segs = segments[1:]
    for i, seg in enumerate(chapter_segs):
        content = strip_bab_lines(seg["content"])
        if i == len(chapter_segs) - 1:
            bm = BACKMATTER_RE.search(content)
            if bm:
                content = content[:bm.start()]
        if len(content.split()) < 40 or is_toc_like(content):
            continue
        raw.append({"title": norm_title(seg["title"]), "content": content})

    if not raw:
        raw = [{"title": "Dokumen", "content": text}]

    chapters = [build_chapter(seg["title"], seg["content"], counter, nodes) for seg in raw]
    return chapters, nodes