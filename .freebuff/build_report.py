# -*- coding: utf-8 -*-
"""
Reformat 'Workshop_Report.pdf' (Deepfake & Digital Forensics) into the exact
format of '5 Days Cybersecurity Bootcamp.pdf':

  * A4, 1-inch margins, Aptos 12pt, bold numbered section headings
  * p1: big bold centered title + centered CyberCell logo + Basic Details
        (Organizing Body/Club: COEP CyberCell in collaboration with ANRF)
  * numbered sections 1..11, photographs grid, Conclusion, Prepared By block
Output: Documents/Workshop_Report_Formatted.pdf  (7 pages)
"""
import os, io, sys, glob
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pymupdf

DOCS = r"C:\Users\Rishab Sen\Documents"
OUT = os.path.join(DOCS, "Workshop_Report.pdf")            # final file (user asked to edit this)
OUT_ALT = os.path.join(DOCS, "Workshop_Report_Formatted.pdf")  # convenience copy
_HERE = os.path.dirname(os.path.abspath(__file__))
LOGO = os.path.join(_HERE, "imgs", "B_p1_x33.jpeg")

W, H = 595.3, 841.9           # A4 portrait
M = 72.0                      # 1 inch margins
RIGHT = 523.3                 # content right edge
BOTTOM = 775.0                # last usable baseline
GRAY1, GRAY2 = (0.63, 0.63, 0.63), (0.89, 0.89, 0.89)

# ---------------- fonts (embedded Aptos extracted from the reference PDF) ----
FDIR = ".freebuff/fonts"
os.makedirs(FDIR, exist_ok=True)
ref = pymupdf.open(os.path.join(DOCS, "5 Days Cybersecurity Bootcamp.pdf"))
names = {}
for xref, ext, ftype, name, ref_, enc in ref.get_page_fonts(0):
    info = ref.extract_font(xref)
    if info[-1]:
        p = os.path.join(FDIR, f"{name}.ttf")
        with open(p, "wb") as f:
            f.write(info[-1])
        names[name] = p
ref.close()
_reg = next(v for k, v in names.items() if k.endswith("+Aptos") or k == "Aptos")
_bold = next(v for k, v in names.items() if "Aptos,Bold" in k)
_ital = next((v for k, v in names.items() if "Aptos,Italic" in k), _reg)
F_REG = pymupdf.Font(fontfile=_reg)
F_BOLD = pymupdf.Font(fontfile=_bold)
F_ITAL = pymupdf.Font(fontfile=_ital)
print("fonts:", sorted(names))

LEAD = 1.33                   # line-height multiple


def san(s):
    """Replace characters whose glyphs are missing from the subset fonts."""
    repl = {"\u2014": "-", "\u2013": "-", "\u2018": "'", "\u2019": "'",
            "\u201c": '"', "\u201d": '"', "\u2026": "..."}
    out = []
    for ch in s:
        if not F_REG.has_glyph(ord(ch)) and ch in repl:
            out.append(repl[ch])
        else:
            out.append(ch)
    return "".join(out)

# ---------------- flow controller -------------------------------------------
class Flow:
    def __init__(self):
        self.pdf = pymupdf.open()
        self.page = None
        self.tw = None
        self.y = M
        self.new_page()

    def new_page(self):
        # flush any pending text on the current page before abandoning it
        if self.page is not None and self.tw is not None and self.tw.last_point:
            self.flush_page()
        self.page = self.pdf.new_page(width=W, height=H)
        self.tw = pymupdf.TextWriter(self.page.rect)
        self.y = M

    def ensure(self, needed):
        if self.y + needed > BOTTOM:
            self.flush_page()
            self.new_page()

    def flush_page(self):
        if self.tw and self.tw.last_point:
            self.tw.write_text(self.page, color=(0, 0, 0))
            self.tw = None

    # ---- primitives ----
    def text(self, x, y, s, size=12, font=None):
        self.tw.append((x, y), san(s), font=font or F_REG, fontsize=size)

    def wrap(self, s, size, width, font=None):
        font = font or F_REG
        out, cur = [], ""
        for w in s.split():
            t = (cur + " " + w).strip()
            if font.text_length(t, fontsize=size) <= width:
                cur = t
            else:
                if cur:
                    out.append(cur)
                cur = w
        if cur:
            out.append(cur)
        return out

    def para(self, s, size=12, italic=True, bold=False, indent=0.0, gap=0):
        font = F_ITAL if italic else (F_BOLD if bold else F_REG)
        self.ensure(size * LEAD)
        width = RIGHT - M - indent
        for ln in self.wrap(s, size, width, font):
            self.ensure(size * LEAD)
            self.text(M + indent, self.y + size, ln, size, font)
            self.y += size * LEAD
        self.y += gap
        return self.y

    def heading(self, s, size=12, gap_before=10, rule_above=False):
        self.y += gap_before
        if rule_above:
            self.rule(gap=2)
        self.ensure(size * LEAD + 6)
        self.text(M, self.y + size, s, size, F_BOLD)
        self.y += size * LEAD
        return self.y

    def bullet(self, s, size=11.5, bold_label=None, indent=18):
        bx = M + indent
        self.ensure(size * LEAD)
        # drawn bullet dot (extracted font subsets lack the \u2022 glyph)
        self.page.draw_circle((M + 6, self.y + size - size * 0.32), 1.4,
                              color=None, fill=(0, 0, 0))
        if bold_label:
            label = san(bold_label) + " "
            lw = F_BOLD.text_length(label, fontsize=size)
            self.text(bx, self.y + size, label, size, F_BOLD)
            for i, ln in enumerate(self.wrap(s, size, RIGHT - bx - lw, F_REG)):
                self.ensure(size * LEAD)
                self.text(bx + (lw if i == 0 else 0), self.y + size, ln, size, F_REG)
                self.y += size * LEAD
        else:
            for ln in self.wrap(s, size, RIGHT - bx, F_REG):
                self.ensure(size * LEAD)
                self.text(bx, self.y + size, ln, size, F_REG)
                self.y += size * LEAD
        return self.y

    def rule(self, gap=8):
        self.ensure(14)
        for yy, col in ((self.y, GRAY1), (self.y + 1, GRAY2)):
            self.page.draw_rect(pymupdf.Rect(M, yy, RIGHT, yy + 0.8), color=None, fill=col)
        self.y += 14 + gap

    def speaker_table(self, rows, col2=95.0, size=11.5):
        self.ensure(len(rows) * size * LEAD + 6)
        for k, v in rows:
            self.text(M, self.y + size, san(k) + " ", size, F_BOLD)
            vlines = self.wrap(v, size, RIGHT - M - col2, F_REG)
            for i, ln in enumerate(vlines):
                self.text(M + col2, self.y + size + i * size * LEAD, ln, size, F_REG)
            self.y += size * LEAD * max(1, len(vlines))
        self.y += 6

    def image_row(self, paths, box_w, box_h, gap=12):
        """Center a row of images scaled to fit (box_w x box_h) each."""
        n = len(paths)
        total = n * box_w + (n - 1) * gap
        x = M + max(0, (RIGHT - M - total) / 2)
        for p in paths:
            img = pymupdf.Pixmap(p)
            ar = img.height / img.width
            w = box_w
            h = min(box_h, w * ar)
            if h < w * ar:
                w = h / ar
            r = pymupdf.Rect(x, self.y, x + w, self.y + h)
            self.page.insert_image(r, filename=p)
            x += box_w + gap
        self.y += max(box_h for _ in paths) + 10

    def centered(self, s, size=11, bold=True, gap=0):
        font = F_BOLD if bold else F_REG
        w = font.text_length(s, fontsize=size)
        self.ensure(size * LEAD + gap)
        self.text((W - w) / 2, self.y + size, s, size, font)
        self.y += size * LEAD + gap


def rotate_jpeg(src, dst):
    """Bake camera-orientation (EXIF) into the pixels; phones store rotated EXIF."""
    from PIL import Image, ImageOps
    im = Image.open(src)
    im = ImageOps.exif_transpose(im)
    im.save(dst, quality=90)

# ================= CONTENT ====================================================
ORG = "COEP CyberCell in collaboration with ANRF"
flow = Flow()

# ---------- PAGE 1: title + logo + sections 1-3 ----------
y = 96.0
for s in ("Deepfake Detection &", "Digital Forensics Workshop Report"):
    w = F_BOLD.text_length(s, fontsize=31)
    flow.text((W - w) / 2, y + 31, s, 31, F_BOLD)
    y += 31 * 1.35
# logo
logo_w = 168.0
r = pymupdf.Rect((W - logo_w) / 2, y + 6, (W + logo_w) / 2, y + 6 + logo_w)
flow.page.insert_image(r, filename=LOGO)
flow.y = r.y1 + 26
flow.rule(gap=4)

flow.heading("1. Basic Details", gap_before=4)
flow.bullet("Deepfake Detection & Digital Forensics Workshop", bold_label="Name of Event/Activity:")
flow.bullet(ORG, bold_label="Organizing Body/Club:")
flow.bullet("3-Day Workshop (Day 1 – Day 3)", bold_label="Date/Duration:")
flow.bullet("COEP Technological University, Pune", bold_label="Venue:")
flow.bullet("Dr. Rajendra Dakhale, Shweta Chawla, Chetan Kawley, Nikhil Birla, "
            "Tejas Limaye (Day 1); Tanmay Dixit (Day 2); Rahul Iyengar, Fardeen Pathan (Day 3)",
            bold_label="Guest(s)/Speaker(s):")

flow.heading("2. Objective of the Event", rule_above=True)
flow.para("The objective of this workshop was to give participants a complete understanding of "
          "deepfakes as an emerging cyber threat — from how synthetic-media attacks are carried out, "
          "to how such incidents are investigated and prosecuted, and finally to the advanced "
          "machine-learning research that makes detection systems robust and reliable.")

flow.heading("3. Description of the Event", rule_above=True)
flow.para("The 3-day Deepfake Detection & Digital Forensics Workshop was organized by " + ORG +
          ". The sessions combined real-world case studies with technical concepts, investigative "
          "methods and research-oriented discussions. Day 1 introduced the deepfake landscape, "
          "attacks, real-world incidents and forensic investigation; Day 2 was a hands-on practical "
          "session on deepfake and cybercrime investigation; and Day 3 focused on advanced ML "
          "research for forensic analysis, generalization, multimodal forensics and authentication "
          "of digital evidence.")

# ---------- PAGE 2: sections 4-8 ----------
flow.heading("4. Activities Conducted", gap_before=14, rule_above=True)
for a in [
    "Speaker-wise sessions on the deepfake threat landscape, attacks and real-world incidents",
    "Case-study reconstruction of documented deepfake incidents (Hong Kong 2024, Bengaluru 2024)",
    "Hands-on cybercrime and deepfake investigation practical with forensic tools",
    "Demonstrations of image-forensics and AI-based detection techniques",
    "Interactive discussions, doubt-solving and research Q&A with the speakers",
]:
    flow.bullet(a)

flow.heading("5. Participation Details", rule_above=True)
flow.bullet("50+", bold_label="Total Number of Participants:")
flow.bullet("Students and enthusiasts interested in cybersecurity, AI and digital forensics",
            bold_label="Target Audience:")
flow.bullet("Domain experts from industry, investigation practice and academia",
            bold_label="Guest(s)/Speaker(s):")

flow.heading("6. Outcomes and Impact", rule_above=True)
flow.para("Participants developed a clear picture of the deepfake deception ecosystem and a working "
          "methodology for investigating manipulated media: verify the source, combine multiple "
          "forensic indicators, and treat online artifacts as evidence. The workshop connected "
          "practical investigation with current ML research — domain adaptation, generalization, "
          "multimodal analysis and out-of-distribution detection — encouraging participants to "
          "explore advanced forensic AI.")

flow.heading("7. Challenges Faced (if any)", rule_above=True)
flow.para("Covering the breadth of concepts, tools and research directions within the limited "
          "session time was challenging, and some hands-on segments had to be compressed; the "
          "speakers handled this efficiently with focused demonstrations.")

flow.heading("8. Learnings and Recommendations", rule_above=True)
flow.para("The workshop emphasized that deepfake detection is a multi-signal, evidence-driven "
          "discipline rather than a single-model problem. It is recommended to organize follow-up "
          "hands-on labs on forensic tooling and Indian-dataset benchmarks so that participants can "
          "apply these concepts end-to-end.")

# ---------- Section 9: day-wise session analysis (pages 3-5) ----------
flow.heading("9. Day-wise Session Analysis", gap_before=16, rule_above=True)

def session(title, day, speaker, time, focus, overview, points, takeaway, size=11):
    flow.ensure(80)
    flow.text(M, flow.y + size + 1, title, size + 1, F_BOLD)
    flow.y += (size + 1) * LEAD + 4
    flow.speaker_table([("Day", day), ("Speaker(s)", speaker), ("Time", time), ("Focus", focus)], size=size)
    flow.para("Session overview: " + overview, size=size)
    flow.heading("Key discussion points", size=size, gap_before=4)
    for b in points:
        flow.bullet(b, size=size)
    flow.heading("Key learning / analytical takeaway", size=size, gap_before=4)
    flow.para(takeaway, size=size, gap=4)

def day_overall(day_label, bullets):
    flow.ensure(60)
    flow.text(M, flow.y + 12, day_label, 12, F_BOLD)
    flow.y += 12 * LEAD
    for b in bullets:
        flow.bullet(b, size=11)

flow.heading("9.1 Day 1 – Understand the Threat", size=12.5, gap_before=6)
flow.para("Day 1 established the foundation for understanding deepfakes and their risks, moving from "
          "high-level concepts to investigation and forensic examination of manipulated media.")

session("Session 1: Deepfake Landscape and Real-World Impact",
        "Day 1", "Dr. Rajendra Dakhale", "11:10 am – 12:15 pm",
        "Deepfake Landscape and Real-World Impact",
        "The first session introduced deepfakes through real-world examples and examined why synthetic "
        "media has become a significant security and social concern.",
        ["How deepfake technology has developed and how attacks can be carried out.",
         "Real-world deepfake examples and the financial or reputational harm that can result.",
         "The role of phishing and social engineering in deepfake-enabled attacks.",
         "Audio deepfakes vs visual manipulation; evolution of techniques and the threat landscape.",
         "Foundations of deepfake detection and AI-based detection approaches.",
         "Threats and potential use of AI-generated/deepfake content for deception."],
        "Deepfakes should be viewed as part of a wider deception ecosystem: detection must consider "
        "not only visual artifacts but also audio, context, attack strategy and the way manipulated "
        "content is delivered to a target.")

session("Session 2: Deepfake Attacks, Identification and Digital Evidence",
        "Day 1", "Shweta Chawla", "1:35 pm – 2:45 pm",
        "Deepfake Attacks, Identification and Digital Evidence",
        "This session focused on how deepfake attacks can be understood through their observable "
        "patterns, indicators and supporting digital evidence.",
        ["Identification of different types of deepfake attacks and attack patterns.",
         "How manipulated media may be used in fraudulent scenarios.",
         "Visual clues and supporting evidence for identifying suspicious content.",
         "Importance of the context in which an image or media file appears.",
         "Investigating content rather than treating a single visual clue as conclusive.",
         "Relationship between deepfake detection, digital forensics and cybercrime investigation."],
        "Deepfake identification is strongest when multiple pieces of evidence are combined. A forensic "
        "workflow should establish provenance, context, indicators of manipulation and supporting "
        "evidence rather than relying on one detection signal.")

session("Session 3: Identifying Leaks and Possible Attack Vectors",
        "Day 1", "Chetan Kawley", "3:00 pm",
        "Identifying Leaks and Possible Attack Vectors",
        "The third session introduced an investigative perspective on identifying weaknesses, leaks "
        "and possible attack paths that can support deepfake or cyber incidents.",
        ["Identification of possible leaks and their relevance to an investigation.",
         "Tracing possible attack vectors and how information may be exposed.",
         "Connecting available evidence with the likely source or pathway of an incident."],
        "Tracing an incident back to its source, and understanding the pathway through which "
        "information or media became available to an attacker, is central to investigation.")

session("Session 4: Deepfake Investigation and Digital Forensics",
        "Day 1", "Nikhil Birla; Tejas Limaye", "Session 4",
        "Deepfake Investigation and Digital Forensics",
        "The fourth session used a case-study-driven approach: Nikhil Birla discussed the "
        "investigator's perspective, while Tejas Limaye focused on forensic examination of evidence.",
        ["Case-study-based investigation of a suspected deepfake incident.",
         "Investigator's role: reconstructing events and examining how the attack was carried out.",
         "Forensic examiner's role: examining digital evidence and identifying signs of manipulation.",
         "Revisiting the Hong Kong 2024 incident and how it could be investigated and reconstructed.",
         "Signs of fake images: lighting mismatch, boundary artifacts, blinking/gaze inconsistencies.",
         "Verifying source and provenance rather than relying only on pixel-level analysis.",
         "Defence scenarios (July 2024 examples) closed through evidence-based reasoning."],
        "Forensic deepfake analysis should move beyond 'does the image look fake?' to 'where did it "
        "come from, how was it created or altered, what evidence supports the conclusion, and can the "
        "incident be reconstructed?'")

day_overall("Day 1 – Overall Analysis", [
    "Day 1 progressed from the broad threat of deepfakes to practical investigation and forensic examination.",
    "Deepfake detection is not purely a computer-vision problem; it is also an investigation and evidence-verification problem.",
    "Real incidents (Hong Kong 2024) connected technical detection with actual attack scenarios.",
    "The investigator/forensic-examiner distinction requires both incident reconstruction and technical evidence analysis.",
])

flow.heading("9.2 Day 2 – Investigate the Incident", size=12.5, gap_before=10)
flow.para("Day 2 focused on practical cybercrime and deepfake investigation, connecting suspicious "
          "online content with evidence collection, technical analysis and investigative tools.")

session("Session 1: Practical Deepfake and Cybercrime Investigation",
        "Day 2", "Tanmay Dixit", "10:20 am – 1:00 pm",
        "Practical Deepfake and Cybercrime Investigation",
        "The session used practical scenarios to demonstrate how investigators can approach "
        "suspicious digital content and online incidents.",
        ["A lost-phone scenario and use of Google/digital services to locate relevant information.",
         "Use of available online information and evidence during an investigation.",
         "Fake websites and suspicious domains as sources of fraud or deception.",
         "Identification and analysis of IP addresses associated with suspicious activity.",
         "Tracking links to identify the IP address or technical origin of an incident.",
         "Cryptographic concepts and encryption/decryption in digital investigations.",
         "Image-analysis and AI-based forensic tools; practical deepfake-investigation takeaways."],
        "A hands-on investigative mindset: identify the incident, preserve clues, trace technical "
        "indicators and use appropriate forensic tools — combining open-source information, network "
        "indicators, cryptographic concepts and media analysis.")

day_overall("Day 2 – Overall Analysis", [
    "Day 2 shifted the workshop toward practical investigation and tool-oriented thinking.",
    "Ordinary online artifacts — links, websites, domains, IP addresses, images, devices — can become evidence.",
    "Deepfake investigation was connected to the broader field of cyber forensics, not treated as an isolated AI problem.",
    "The practical orientation made the concepts applicable to real incident-response and digital-evidence workflows.",
])

flow.heading("9.3 Day 3 – Build Better Systems (Research Directions)", size=12.5, gap_before=10)
flow.para("Day 3 moved toward advanced research: machine learning for forensic applications, "
          "generalization across domains and datasets, multimodal analysis, robustness, "
          "explainability and out-of-distribution detection.")

session("Session 1: Advanced Machine Learning for Forensic Analysis",
        "Day 3", "Rahul Iyengar", "10:30 am",
        "Advanced Machine Learning for Forensic Analysis",
        "The session explored advanced research directions for applying machine learning and deep "
        "learning to forensic problems.",
        ["Shortcut learning in forensic CNNs; attention and noise learning.",
         "Domain-incremental learning (incl. CVPR 2026 research); biometric facial manipulation analysis.",
         "Fourier spectral power-law invariants; audio-visual multimodal learning.",
         "Domain-adaptive methods; optimal transport and Earth Mover's Distance; source-free adaptation.",
         "Domain generalization, invariant representations and meta-learning.",
         "Closed-set vs open-set recognition; openness metrics, open-space risk, energy-based OOD scoring.",
         "Diffusion and flow matching in forensics; spatiotemporal detection in video.",
         "SOTA benchmark comparison, cross-dataset generalization, explainability and saliency."],
        "The central research message was robustness and generalization: a forensic AI model should "
        "learn meaningful forensic evidence, generalize across domains and sources, recognize "
        "unfamiliar cases and provide interpretable evidence for its decisions.")

session("Session 2: Multimodal Forensics, Indian Datasets and Advanced Detection",
        "Day 3", "Fardeen Pathan", "12:45 pm",
        "Multimodal Forensics, Indian Datasets and Advanced Detection",
        "The second Day 3 session continued the research focus with multimodal systems, Indian "
        "datasets, model architectures, authentication, fusion and cryptographic techniques.",
        ["Bengaluru 2024 incident as a forensic reference; fast semantic and frequency-based analysis.",
         "Need for Indian datasets for developing and evaluating forensic systems.",
         "ViT, DINOv2, SRM and DCT/FFT-related approaches; 32-channel feature representations.",
         "EfficientNet and related architectures; loss functions and authentication/fusion approaches.",
         "PAC, Merkle trees and Falcon-512 in the context of security and verification.",
         "Cryptographic algorithms and their relevance to trustworthy digital evidence."],
        "Forensic systems need both strong detection models and trustworthy mechanisms for "
        "authenticating and validating digital evidence — combining locally relevant datasets, "
        "semantic and frequency information, and integrated security mechanisms.")

day_overall("Day 3 – Overall Analysis", [
    "Day 3 represented a transition from applied investigation to advanced research methodology.",
    "Generalization, domain adaptation, multimodal learning and OOD detection emerged as major themes for reliable forensic AI.",
    "The Indian-datasets discussion stressed evaluating models on data reflecting the actual deployment environment.",
    "Explainability and source verification continued the investigative lessons of Days 1-2 across the workshop.",
])

# ---------- Section 10: photographs + conclusion + prepared-by (final page) ----------
flow.new_page()
flow.heading("10. Photographs", gap_before=0, rule_above=True)
flow.para("Glimpses from the three days of the workshop - sessions, hands-on lab work, "
          "dais proceedings and the closing group photograph with speakers and organizers.", size=11, gap=4)
SRC = r"C:\Users\Rishab Sen\Downloads"
photo_files = [
    "20260831_104358.jpg",              # Day 1 audience in the computer lab
    "20260831_110812AMByGPSMapCamera.jpg",  # Day 1 auditorium session
    "20260831_134940.jpg",              # Day 1 speaker at the dais
    "20260831_135155.jpg",              # Day 1 dais with guests
    "20260831_164513.jpg",              # Day 1 projection screens
    "20260831_172403.jpg",              # Day 1 lab practical
    "20260901_120612.jpg",              # Day 2 hands-on investigation lab
    "20260901_164803.jpg",              # Day 2 session
    "20260901_44616PMByGPSMapCamera.jpg",   # Day 3 research session
    "20260902_104859.jpg",              # closing group photograph
]
from PIL import Image, ImageOps, ImageEnhance
enh = []
for i, name in enumerate(photo_files):
    rot = os.path.join(_HERE, "render", f"rot_{i}.jpg")
    rotate_jpeg(os.path.join(SRC, name), rot)
    im = Image.open(rot).convert("RGB")
    im.thumbnail((1100, 1100))
    im = ImageOps.autocontrast(im, cutoff=1)
    im = ImageEnhance.Brightness(im).enhance(1.08)
    im = ImageEnhance.Contrast(im).enhance(1.12)
    ep = os.path.join(_HERE, "render", f"photo_{i}.jpg")
    im.save(ep, quality=86)
    enh.append(ep)

PB_CAP = 10   # caption font size
def photo_row(paths, caption, max_h=175, gap=10):
    """Justified row: all photos share one height, row spans the content width."""
    avail = RIGHT - M
    ars = []
    for p in paths:
        im = Image.open(p)
        ars.append(im.width / im.height)
    h = (avail - gap * (len(paths) - 1)) / sum(ars)
    h = min(h, max_h)
    total = h * sum(ars) + gap * (len(paths) - 1)
    x = M + max(0, (avail - total) / 2)
    flow.ensure(h + PB_CAP + 20)
    top = flow.y
    for p, ar in zip(paths, ars):
        w = h * ar
        flow.page.insert_image(pymupdf.Rect(x, top, x + w, top + h), filename=p)
        x += w + gap
    flow.y = top + h + 6
    fw = F_ITAL.text_length(caption, fontsize=PB_CAP)
    flow.text((W - fw) / 2, flow.y + PB_CAP, caption, PB_CAP, F_ITAL)
    flow.y += PB_CAP + 12

photo_row(enh[0:3], "Day 1 - Audience and speaker sessions")
photo_row(enh[3:6], "Day 1 - Dais proceedings and projection screens")
photo_row(enh[6:8], "Day 2 - Hands-on investigation lab", max_h=170)
# page break so the last two photos share the final page with the conclusion
flow.flush_page()
flow.new_page()
flow.heading("10. Photographs (continued)", gap_before=0, rule_above=True)
flow.y += 8
photo_row(enh[8:10], "Day 3 research session and closing group photograph", max_h=165)

# ---------- Section 11: conclusion + prepared by ----------
flow.heading("11. Conclusion", gap_before=6, rule_above=True)
flow.para("Across the three days, the workshop presented a complete perspective on deepfake and "
          "digital-forensic challenges - from understanding how attacks occur, to investigating real "
          "incidents, examining evidence and designing more robust AI systems. The speaker sessions "
          "collectively demonstrated that reliable deepfake forensics requires a combination of "
          "investigation, source verification, computer vision, multimodal learning, cybersecurity, "
          "cryptography and careful evaluation of model robustness.", size=11)
flow.para("The overall learning can be summarized as: detect the manipulation, verify the source, "
          "reconstruct the incident, validate the evidence, and design AI systems that remain "
          "reliable when the data, domain or attack method changes.", size=11)

pb = 12.5
flow.y += 8
flow.text(M, flow.y + pb, "Prepared By:", pb, F_BOLD)
flow.text(M + 92, flow.y + pb, "Rishab Sen", pb, F_REG)
flow.y += pb * LEAD + 2
flow.text(M, flow.y + pb, "Designation:", pb, F_BOLD)
flow.text(M + 92, flow.y + pb, "Member", pb, F_REG)
flow.y += pb * LEAD + 2
flow.text(M, flow.y + pb, "Date of Submission:", pb, F_BOLD)
flow.text(M + 122, flow.y + pb, "September 23, 2026", pb, F_REG)
flow.y += pb * LEAD + 10

flow.rule(gap=6)
flow.centered("Organized by " + ORG, size=11.5, bold=True)

# ---------- save ----------
flow.flush_page()
flow.pdf.set_metadata({
    "title": "Deepfake Detection & Digital Forensics Workshop Report",
    "author": ORG,
    "creator": "COEP CyberCell",
})
flow.pdf.save(OUT, garbage=4, deflate=True)
flow.pdf.save(OUT_ALT, garbage=4, deflate=True)
print("saved:", OUT, "| pages:", len(flow.pdf))
for i, page in enumerate(flow.pdf):
    print(f"  page {i+1}: {len(page.get_text()):>5} chars, {len(page.get_images())} images")
