# -*- coding: utf-8 -*-
"""v3: multi-route OA PDF fetcher: Unpaywall -> Semantic Scholar -> Wayback -> EPMC XML."""
import os, io, json, re, time, tarfile
import urllib.request, urllib.parse

OUT = r"C:\Users\Rishab Sen\Documents\Research Papers"
os.makedirs(OUT, exist_ok=True)

H = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36",
     "Accept": "*/*"}
EMAIL = "mmr.seminar.research@coeptech.ac.in"

PAPERS = [
    dict(n=1,  key="01_Iddan_2000_Nature_WirelessCapsuleEndoscopy", doi="10.1038/35013140", pmcid=None),
    dict(n=2,  key="02_Melzer_2012_Lancet_PhysicsMedicalTreatment", doi="10.1016/S0140-6736(12)60428-0", pmcid=None),
    dict(n=3,  key="03_Xiao_2021_LancetGAH_MagneticallyControlledCE", doi="10.1016/S2468-1253(21)00274-0", pmcid=None),
    dict(n=4,  key="04_Bladen_1993_Lancet_3DEndoscopeImaging", doi="10.1016/0140-6736(93)90487-2", pmcid=None),
    dict(n=5,  key="05_Shah_2000_Lancet_MagneticEndoscopyImaging", doi="10.1016/S0140-6736(00)03205-0", pmcid=None),
    dict(n=6,  key="06_Gora_2013_NatMed_TetheredCapsuleEndomicroscopy", doi="10.1038/nm.3052", pmcid="PMC3567218"),
    dict(n=7,  key="07_Liang_2020_BMJOG_TetheredCapsuleOCT", doi="10.1136/bmjgast-2020-000444", pmcid="PMC7473663"),
    dict(n=8,  key="08_Yang_2012_NatMed_PhotoacousticUltrasonicEndoscopy", doi="10.1038/nm.2823", pmcid="PMC3885361"),
    dict(n=9,  key="09_Abramson_2019_NatMed_LuminalMicroneedleInjector", doi="10.1038/s41591-019-0598-9", pmcid="PMC7218658"),
    dict(n=10, key="10_Xie_2022_JAMA_AI_CapsuleVideoReview", doi="10.1001/jamanetworkopen.2022.21992", pmcid="PMC9284338"),
    dict(n=11, key="11_Morgan_2016_BMJOG_CapsuleColonoscopy", doi="10.1136/bmjgast-2016-000089", pmcid="PMC4860721"),
    dict(n=12, key="12_Vilz_2016_BMJOpen_SmartPill_Ileus", doi="10.1136/bmjopen-2015-011014", pmcid="PMC4947765"),
]


def fetch(url, timeout=45, headers=H):
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def is_pdf(b):
    return b[:5] == b"%PDF-"


def route_unpaywall(doi):
    u = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi, safe='')}?email={EMAIL}"
    try:
        j = json.loads(fetch(u))
    except Exception as e:
        print("   A unpaywall:", e)
        return None
    seen, cands = set(), []
    loc = j.get("best_oa_location") or {}
    for L in [loc] + j.get("oa_locations", []):
        for k in ("url_for_pdf", "url"):
            v = L.get(k)
            if v and v not in seen and "/fullTextXML" not in v:
                seen.add(v); cands.append(v)
    for c in cands:
        try:
            b = fetch(c)
            if is_pdf(b):
                return b, c
        except Exception as e:
            print("   A fail:", str(e)[:70])
    return None


def route_semanticscholar(doi):
    u = (f"https://api.semanticscholar.org/graph/v1/paper/DOI:{urllib.parse.quote(doi, safe='')}"
         f"?fields=openAccessPdf")
    try:
        j = json.loads(fetch(u))
    except Exception as e:
        print("   B s2:", e)
        return None
    pdf = (j.get("openAccessPdf") or {}).get("url")
    if not pdf:
        return None
    for cand in [pdf, pdf.replace("http://", "https://")]:
        try:
            b = fetch(cand)
            if is_pdf(b):
                return b, cand
        except Exception as e:
            print("   B fail:", str(e)[:70])
    return None


def route_wayback(pmcid):
    """Look for archived PMC PDF snapshots."""
    patterns = [
        f"pmc.ncbi.nlm.nih.gov/articles/{pmcid}/pdf/*",
        f"www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/*",
    ]
    for pat in patterns:
        u = (f"https://web.archive.org/cdx/search/cdx?url={urllib.parse.quote(pat)}"
             f"&output=json&filter=statuscode:200&limit=5")
        try:
            rows = json.loads(fetch(u, timeout=60))
        except Exception as e:
            print("   C cdx:", str(e)[:70])
            continue
        for row in rows[1:]:
            ts, orig = row[1], row[2]
            try:
                b = fetch(f"https://web.archive.org/web/{ts}id_/{orig}", timeout=90)
                if is_pdf(b):
                    return b, f"wayback:{orig}"
            except Exception as e:
                print("   C fail:", str(e)[:70])
    return None


XML2PDF = """
Converts Europe PMC open-access full-text XML into a clean, readable PDF
(using reportlab). Used only when the publisher PDF cannot be fetched.
"""
def route_epmc_xml_pdf(doi, key):
    q = urllib.parse.quote(f'DOI:"{doi}"')
    u = f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query={q}&format=json&resultType=core"
    try:
        res = json.loads(fetch(u)).get("resultList", {}).get("result", [])
        if not res:
            return None
        pmcid = res[0].get("pmcid")
        if not pmcid:
            return None
        xml = fetch(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
    except Exception as e:
        print("   D epmc:", str(e)[:70])
        return None
    try:
        import xml.etree.ElementTree as ET
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import mm
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        print("   D reportlab missing")
        return None

    root = ET.fromstring(xml)
    texts = []

    def grab(el):
        return re.sub(r"<[^>]+>", "", ET.tostring(el, encoding="unicode", method="html")
                      ) if el is not None else ""

    art = root.find(".//article") if root.find(".//article") is not None else root
    title = ""
    for t in art.iter("article-title"):
        title = "".join(t.itertext()); break
    authors = []
    for a in art.iter("name"):
        ln = a.findtext("surname"); ini = "".join(x.text or "" for x in a.findall("initials"))
        if ln: authors.append(f"{ini} {ln}")
    journal = art.findtext(".//journal-title") or ""
    year = art.findtext(".//year") or ""
    abs_texts = []
    for ab in art.iter("abstract"):
        abs_texts.append(" ".join("".join(x.itertext()) for x in ab.iter("p")))
    body = []
    for sec in art.iter("sec"):
        sec_title = sec.findtext("title") or ""
        sec_text = " ".join("".join(x.itertext()) for x in sec.iter("p"))
        if sec_text.strip():
            body.append((sec_title, sec_text))

    styles = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=styles["Title"], fontSize=14, leading=18)
    meta = ParagraphStyle("meta", parent=styles["Normal"], fontSize=10, leading=13,
                          textColor="#444444")
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, leading=15)
    norm = ParagraphStyle("norm", parent=styles["Normal"], fontSize=9.5, leading=13,
                          spaceAfter=6)

    buf_path = os.path.join(OUT, key + ".pdf")
    doc = SimpleDocTemplate(buf_path, pagesize=A4, title=title[:120],
                            leftMargin=18*mm, rightMargin=18*mm,
                            topMargin=16*mm, bottomMargin=16*mm)
    story = [Paragraph(title or "(no title)", h1), Spacer(1, 4),
             Paragraph(f"{', '.join(authors[:12])}", meta),
             Paragraph(f"{journal} ({year}) - DOI: {doi} - PMC: {pmcid}", meta),
             Spacer(1, 6),
             Paragraph("NOTE: Regenerated for academic reference from the Europe PMC "
                       "open-access full-text XML. For the typeset version of record, "
                       "see the DOI link.", meta),
             Spacer(1, 10)]
    if abs_texts:
        story.append(Paragraph("Abstract", h2)); story.append(Spacer(1, 3))
        for a in abs_texts:
            story.append(Paragraph(a[:4000], norm))
    for st, tx in body[:60]:
        if st:
            story.append(Paragraph(st, h2))
        story.append(Paragraph(tx[:4000], norm))
    doc.build(story)
    data = open(buf_path, "rb").read()
    return (data, "epmc-xml-regenerated") if is_pdf(data) else None


results = []
for p in PAPERS:
    print(f"\n[{p['n']}/12] {p['key']}")
    data = src = None

    r = route_unpaywall(p["doi"])
    if r: data, src = r; print("   A OK:", src[:60])

    if not data:
        r = route_semanticscholar(p["doi"])
        if r: data, src = r; print("   B OK:", src[:60])

    if not data and p["pmcid"]:
        r = route_wayback(p["pmcid"])
        if r: data, src = r; print("   C OK:", src[:60])

    if not data:
        r = route_epmc_xml_pdf(p["doi"], p["key"])
        if r: data, src = r; print("   D OK (regenerated from OA XML)")

    if data:
        open(os.path.join(OUT, p["key"] + ".pdf"), "wb").write(data)
        print(f"   SAVED ({len(data)//1024} KB)")
        status = "OK-PUBLISHER" if src and "regenerated" not in str(src) and "wayback" not in str(src) else "OK-ARCHIVE/REGEN"
    else:
        status = "PAYWALLED"
        print("   -> PAYWALLED (no legal OA copy)")
    results.append(dict(n=p["n"], key=p["key"], doi=p["doi"], status=status,
                        source=str(src) if src else None, bytes=len(data) if data else 0))
    time.sleep(1.0)

with open(os.path.join(OUT, "_manifest.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n===== FINAL SUMMARY =====")
for r in results:
    print(f"[{r['status']:>18}] {r['key']}")
ok = sum(1 for r in results if r["status"].startswith("OK"))
print(f"\nTotal PDFs: {ok}/12")
