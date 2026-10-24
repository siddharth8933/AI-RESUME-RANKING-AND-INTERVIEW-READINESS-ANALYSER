"""AI Resume Ranking & Interview Readiness Analyzer - backend.
Modules: (1) parsing  (2) TF-IDF + cosine ranking  (3) resume quality  (4) interview readiness.
Sirf standard library (PDF ke liye pypdf). Resume disk par save nahi hota.
Run: python server.py -> http://localhost:8000"""
import base64, io, json, math, os, re, zipfile
from collections import Counter
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# ---------- Skill dictionary: canonical name -> aliases (same skill, alag wording) ----------
SKILLS = {
 "python": ["python"], "java": ["java"], "javascript": ["javascript", "js", "node.js", "nodejs"],
 "c++": ["c++"], "sql": ["sql", "mysql", "postgresql", "sqlite"], "mongodb": ["mongodb"],
 "html/css": ["html", "css", "html5", "css3"], "react": ["react", "react.js"], "flask": ["flask"],
 "django": ["django"], "git": ["git", "github"], "docker": ["docker"], "aws": ["aws", "amazon web services"],
 "machine learning": ["machine learning", "ml", "scikit-learn", "sklearn"],
 "deep learning": ["deep learning", "neural network", "tensorflow", "pytorch"],
 "nlp": ["nlp", "natural language processing", "spacy", "nltk"],
 "data analysis": ["data analysis", "pandas", "numpy", "excel"], "rest api": ["rest", "api", "apis"],
 "communication": ["communication"], "teamwork": ["teamwork", "team player", "collaboration"],
 "leadership": ["leadership", "led a team"], "problem solving": ["problem solving", "problem-solving"],
}
QUESTIONS = {
 "python": "Explain lists vs tuples, and how a Python decorator works.",
 "sql": "Write a query to find the second highest salary. What is a JOIN?",
 "machine learning": "Explain overfitting and two ways to reduce it.",
 "nlp": "What is TF-IDF and why is cosine similarity used with it?",
 "javascript": "What is a closure? How does async/await differ from callbacks?",
 "react": "What are hooks, and when does a component re-render?",
 "flask": "How does a Flask route handle a POST request?",
 "git": "Difference between merge and rebase?",
 "docker": "What is the difference between an image and a container?",
 "data analysis": "How would you handle missing values in a dataset?",
 "java": "Explain OOP pillars with a real example.",
 "deep learning": "What does backpropagation do?",
}
VERBS = "led built created designed developed improved managed launched reduced increased delivered automated analyzed implemented optimized trained wrote deployed".split()
STOP = set("the and for with that this from are was were will have has you your our their can all any not but they them its into who how use using used been being also such than then over per etc job role work team strong good".split())

def has(text, alias):
    return re.search(r"(?<![\w+])" + re.escape(alias) + r"(?![\w+])", text) is not None

def find_skills(low):
    return [s for s, al in SKILLS.items() if any(has(low, a) for a in al)]

# ---------- 1. Parsing ----------
def extract(name, data):
    if name.lower().endswith(".docx"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            xml = z.read("word/document.xml").decode("utf8", "ignore")
        return re.sub(r"<[^>]+>", " ", xml.replace("</w:p>", "\n"))
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ValueError("PDF reading needs pypdf. Run: pip install pypdf (or upload a DOCX).")
    return "\n".join(p.extract_text() or "" for p in PdfReader(io.BytesIO(data)).pages)

def parse(text):
    low = text.lower()
    name = next((l.strip() for l in text.splitlines()
                 if 2 <= len(l.split()) <= 4 and re.fullmatch(r"[A-Za-z. ]+", l.strip())), "Unknown")
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.]+", text)
    phone = re.search(r"\+?\d[\d\s-]{8,}\d", text)
    edu = re.findall(r"\b(b\.?\s?tech|m\.?\s?tech|bca|mca|mba|b\.?\s?sc|m\.?\s?sc|bachelor|master|diploma)\b", low)
    yrs = [float(x) for x in re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)", low)]
    return {"name": name, "email": email.group() if email else "", "phone": phone.group().strip() if phone else "",
            "education": sorted(set(e.replace(" ", "").replace(".", "").upper() for e in edu)),
            "years": max(yrs) if yrs else 0, "skills": find_skills(low),
            "projects": min(5, len(re.findall(r"\bprojects?\b", low))), "words": len(text.split())}

# ---------- 2. TF-IDF + cosine similarity ----------
def tokens(t): return [w for w in re.findall(r"[a-z][a-z+#.]{1,}", t.lower()) if w not in STOP]

def cosine_scores(jd, resumes):
    docs = [Counter(tokens(d)) for d in [jd] + resumes]
    df = Counter(w for d in docs for w in d)
    n = len(docs)
    vec = [{w: c * (math.log((1 + n) / (1 + df[w])) + 1) for w, c in d.items()} for d in docs]
    norm = lambda v: math.sqrt(sum(x * x for x in v.values())) or 1
    return [sum(x * vec[0].get(w, 0) for w, x in v.items()) / (norm(v) * norm(vec[0])) for v in vec[1:]]

def rank(p, jd_skills, cos):
    sim = min(100, cos * 220)  # raw cosine is usually 0.1-0.4, isliye scale kiya
    cov = len([s for s in jd_skills if s in p["skills"]]) / len(jd_skills) * 100 if jd_skills else sim
    prof = (50 if p["education"] else 0) + min(50, p["projects"] * 15 + p["years"] * 10)
    return {"match": round(.5 * sim + .4 * cov + .1 * prof), "similarity": round(sim),
            "coverage": round(cov), "profile": round(prof)}

# ---------- 3. Resume quality ----------
def quality(text, p):
    low, f = text.lower(), []
    add = lambda pr, t, d, fx: f.append({"priority": pr, "title": t, "detail": d, "fix": fx})
    ats = 100
    if not p["email"]: ats -= 30; add("High", "No email found", "Recruiters need a way to reach you.", "Add a professional email in the header.")
    if not p["phone"]: ats -= 15; add("Medium", "No phone number found", "Contact details look incomplete.", "Add a number with country code.")
    if p["words"] < 150: ats -= 25; add("High", "Resume looks too short", f"Only {p['words']} words could be read. It may be a scanned PDF.", "Use a text-based PDF or DOCX.")
    secs = [s for s in ("summary", "experience", "education", "skills", "projects") if s in low]
    for s in {"experience", "education", "skills", "projects"} - set(secs):
        add("Medium", f"No '{s.title()}' section", "Standard headings help parsers and recruiters.", f"Add a clear '{s.title()}' heading.")
    nums = len(re.findall(r"\d+%|\$\d+|\b\d{2,}\b", text))
    verbs = sum(has(low, v) for v in VERBS)
    if nums < 3: add("High", "Few measurable results", "Impact is hard to judge without numbers.", "Example: 'Cut page load time by 40%'.")
    if verbs < 4: add("Medium", "Weak action verbs", "Bullets read like duties, not achievements.", "Start bullets with built, led, improved, automated.")
    issues = len(re.findall(r"\b(\w+) \1\b", low)) + len(re.findall(r"\bi\b(?= [a-z])", text))
    if issues: add("Low", "Small wording slips", f"{issues} repeated words or lowercase 'i' found.", "Proofread once, aloud.")
    cats = {"ATS readability": ats, "Resume sections": len(secs) * 20,
            "Impact and clarity": min(100, nums * 12 + verbs * 8 + 20), "Grammar basics": max(0, 100 - issues * 10)}
    return {"score": round(sum(cats.values()) / 4), "categories": cats, "findings": f}

# ---------- 4. Interview readiness ----------
def readiness(p, jd_skills, q):
    miss = [s for s in jd_skills if s not in p["skills"]]
    skill = (len(jd_skills) - len(miss)) / len(jd_skills) if jd_skills else min(1, len(p["skills"]) / 8)
    soft = sum(s in p["skills"] for s in ("communication", "teamwork", "leadership", "problem solving"))
    parts = {"Skill coverage": skill * 40, "Projects": min(1, p["projects"] / 3) * 20,
             "Experience and impact": min(1, q["categories"]["Impact and clarity"] / 80) * 20,
             "Soft skills": min(1, soft / 2) * 20}
    topics = [{"skill": s, "question": QUESTIONS.get(s, f"Explain a project where you used {s}.")} for s in miss[:5]]
    if not topics:
        topics = [{"skill": s, "question": QUESTIONS[s]} for s in p["skills"] if s in QUESTIONS][:4]
    tips = ["Prepare a 60-second 'Tell me about yourself'.", "Pick 2 projects and be ready to explain every design choice."]
    if soft < 2: tips.append("Add one story each for teamwork and problem solving (STAR format).")
    return {"score": round(sum(parts.values())), "parts": {k: round(v) for k, v in parts.items()}, "missing": miss, "topics": topics, "tips": tips}

# ---------- API ----------
def analyze(files, jd):
    texts, names = [], []
    for f in files:
        if not f["name"].lower().endswith((".pdf", ".docx")): raise ValueError(f"{f['name']}: only PDF or DOCX allowed.")
        texts.append(extract(f["name"], base64.b64decode(f["data"])))
        names.append(f["name"])
    jd_skills = find_skills(jd.lower()) if jd.strip() else []
    cos = cosine_scores(jd, texts) if jd.strip() else [0] * len(texts)
    out = []
    for fn, t, c in zip(names, texts, cos):
        p = parse(t); q = quality(t, p)
        r = rank(p, jd_skills, c) if jd.strip() else None
        if jd_skills and r:
            miss = [s for s in jd_skills if s not in p["skills"]]
            if miss: q["findings"].insert(0, {"priority": "High", "title": "Skills in the job post are missing", "detail": "Missing: " + ", ".join(miss), "fix": "Add them where you honestly have the skill."})
        q["findings"].sort(key=lambda x: ["High", "Medium", "Low"].index(x["priority"]))
        out.append({"file": fn, "profile": p, "quality": q, "rank": r, "readiness": readiness(p, jd_skills, q),
                    "matched": [s for s in jd_skills if s in p["skills"]], "missing": [s for s in jd_skills if s not in p["skills"]]})
    if jd.strip(): out.sort(key=lambda x: -x["rank"]["match"])
    return {"results": out, "hasJD": bool(jd.strip())}

class H(BaseHTTPRequestHandler):
    def send(self, code, body, ct="application/json"):
        self.send_response(code); self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self.send(200, (Path(__file__).parent / "index.html").read_bytes(), "text/html; charset=utf-8")
    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 40 * 1024 * 1024: raise ValueError("Upload too large.")
            req = json.loads(self.rfile.read(n))
            self.send(200, json.dumps(analyze(req["files"], req.get("jd", ""))).encode())
        except ValueError as e:
            self.send(400, json.dumps({"error": str(e)}).encode())
        except Exception:
            self.send(400, json.dumps({"error": "Could not read one of the files. Try another file."}).encode())

if __name__ == "__main__":
    import webbrowser
    if not (Path(__file__).parent / "index.html").exists():
        raise SystemExit("ERROR: index.html is file ke saath, same folder mein honi chahiye.")
    cloud = "PORT" in os.environ  # Render jaisi hosting PORT set karti hai
    host = "0.0.0.0" if cloud else "127.0.0.1"
    for port in ([int(os.environ["PORT"])] if cloud else range(8000, 8011)):  # port busy ho to agla try karo
        try:
            srv = ThreadingHTTPServer((host, port), H); break
        except OSError:
            continue
    else:
        raise SystemExit("ERROR: koi port khali nahi mila. Purane terminal band karo.")
    url = f"http://localhost:{port}"
    print(f"\nWebsite chal rahi hai: {url}\nBand karne ke liye Ctrl+C dabao.\n")
    try:
        import pypdf  # noqa
    except ImportError:
        print("Note: PDF ke liye 'python -m pip install pypdf' chalao. DOCX bina iske chalega.\n")
    if not cloud: webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("Server band.")
