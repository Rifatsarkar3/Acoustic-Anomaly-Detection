"""Convert paper/main.tex to a review-friendly DOCX via Pandoc.

Resolves \\cite{} against the compiled main.bbl (so DOCX reference numbers
match the compiled PDF exactly) into individually bracketed, clickable
citations (each number links to its entry in the References section).
Resolves \\ref{} against the compiled main.aux (LaTeX's own authoritative
label numbering) into clickable cross-references that jump to the target
figure/table/section. Inserts a page break before every top-level section
so the document reads as distinct chapters in Word. Rewrites the elsarticle
frontmatter (author/affiliation/keyword blocks) into plain text pandoc
understands, and appends a manually built References section from main.bbl.

Requires main.bbl and main.aux to exist (compile with `tectonic
--keep-intermediates main.tex` first). Requires pandoc on PATH.

Usage:
    python utils/tex_to_docx.py
"""
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DIR = os.path.join(ROOT, "paper")
TEX_IN = os.path.join(PAPER_DIR, "main.tex")
BBL_IN = os.path.join(PAPER_DIR, "main.bbl")
AUX_IN = os.path.join(PAPER_DIR, "main.aux")
TEX_TMP = os.path.join(PAPER_DIR, "_main_for_docx.tex")
DOCX_OUT = os.path.join(PAPER_DIR, "Acoustic Anomaly manuscript.docx")


def bookmark_name(label):
    """Word bookmark names allow only letters/digits/underscore and must
    start with a letter — sanitize labels like 'sec:intro' or 'fig:pipeline'
    (which contain a colon) rather than relying on pandoc to do it silently.
    """
    name = re.sub(r"[^A-Za-z0-9_]", "_", label)
    if not name[:1].isalpha():
        name = "b_" + name
    return name[:40]


def clean_ref_text(body):
    t = body
    t = re.sub(r"\\end\{thebibliography\}.*", "", t, flags=re.S)
    t = t.replace("\\newblock", " ")
    repl = {
        r'{\"o}': "ö", r'{\"u}': "ü", r"{\'a}": "á", r"{\ss}": "ß",
        r'{\"O}': "Ö", r'{\"U}': "Ü",
    }
    for k, v in repl.items():
        t = t.replace(k, v)
    # \href {URL}\n  {\path{TEXT}} -> TEXT (bbl inserts whitespace/newlines around braces)
    t = re.sub(r"\\href\s*\{[^}]*\}\s*\{\\path\{([^}]*)\}\}", r"\1", t, flags=re.S)
    # strip protective braces around ALL-CAPS acronyms: {MIMII} -> MIMII
    t = re.sub(r"\{([A-Z][A-Z0-9]+)\}", r"\1", t)
    t = t.replace("~", " ")
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("{", "").replace("}", "")
    return t


def main():
    if not os.path.exists(BBL_IN):
        sys.exit(
            "main.bbl not found — compile first:\n"
            "  tectonic --keep-intermediates main.tex   (run from paper/)"
        )
    if not os.path.exists(AUX_IN):
        sys.exit(
            "main.aux not found — compile first:\n"
            "  tectonic --keep-intermediates main.tex   (run from paper/)"
        )

    with open(TEX_IN, encoding="utf-8") as f:
        tex = f.read()
    with open(BBL_IN, encoding="utf-8") as f:
        bbl = f.read()
    with open(AUX_IN, encoding="utf-8") as f:
        aux = f.read()

    entries = re.split(r"\\bibitem\{([^}]+)\}", bbl)[1:]
    keys, bodies = entries[0::2], entries[1::2]
    key_to_num = {k: i + 1 for i, k in enumerate(keys)}
    # Each entry gets a bookmark (\hypertarget) so in-text citations can link to it.
    ref_lines = [
        f"\\hypertarget{{{bookmark_name('cite_' + k)}}}{{}}[{i}] {clean_ref_text(body)}"
        for i, (k, body) in enumerate(zip(keys, bodies), start=1)
    ]
    print(f"Parsed {len(keys)} bibliography entries from main.bbl.")

    def cite_repl(m):
        parts = [
            f"\\hyperlink{{{bookmark_name('cite_' + k.strip())}}}{{[{key_to_num[k.strip()]}]}}"
            for k in m.group(1).split(",")
        ]
        return ", ".join(parts)

    tex = re.sub(r"\\cite\{([^}]+)\}", cite_repl, tex)

    # Resolve \ref{} against LaTeX's own authoritative numbering (main.aux),
    # rather than relying on pandoc's own (incomplete — e.g. it can't see
    # into table* environments) cross-reference tracking. Every number
    # becomes a real hyperlink to a bookmark planted at the matching \label.
    label_to_num = dict(
        re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]*)\}", aux)
    )

    def ref_repl(m):
        label = m.group(1)
        if label not in label_to_num:
            sys.exit(
                f"\\ref{{{label}}} not found in main.aux — recompile main.tex "
                "first, or the label was removed/renamed."
            )
        return f"\\hyperlink{{{bookmark_name(label)}}}{{{label_to_num[label]}}}"

    tex = re.sub(r"\\ref\{([^}]+)\}", ref_repl, tex)

    # Plant a bookmark at every \label{} (figures, tables, sections) so the
    # hyperlinks above have somewhere to land, right at the labelled item.
    #
    # Figure labels are handled separately: a \hypertarget{} placed *inside*
    # a \begin{figure}...\end{figure} float (regardless of exact position)
    # corrupts pandoc's LaTeX reader in a way that doesn't affect that
    # figure itself, but throws off its subsequent table-float parsing for
    # the rest of the document (verified empirically — with this figure-only
    # injection isolated, the docx's <w:tbl> count roughly doubles, 14 -> 25,
    # even though the affected figures aren't tables). Planting the bookmark
    # just after \end{figure*}? instead avoids touching the float's
    # internals and fixes it.
    def fig_env_repl(m):
        body, end_tag = m.group(1), m.group(2)
        label_m = re.search(r"\\label\{(fig:[^}]+)\}", body)
        if not label_m:
            return m.group(0)
        return body + end_tag + f"\\hypertarget{{{bookmark_name(label_m.group(1))}}}{{}}"

    tex = re.sub(
        r"(\\begin\{figure\*?\}.*?)(\\end\{figure\*?\})", fig_env_repl, tex, flags=re.S
    )
    tex = re.sub(
        r"\\label\{((?!fig:)[^}]+)\}",
        lambda m: m.group(0) + f"\\hypertarget{{{bookmark_name(m.group(1))}}}{{}}",
        tex,
    )

    old_front = """\\author[HAU]{Sakar Mohammad Raziul Hasan Rifat}
\\author[HAU]{Akter Labani}
\\author[HAU]{Tao Zhang\\corref{cor1}}
\\ead{taozhang2021@hau.edu.cn}
\\author[HAU]{Saleh Mahamat Aboubakar OUSMANE}
\\author[HAU]{Boudjelkha Mohammed Djamel EDDINE}

\\cortext[cor1]{Corresponding author.}
\\affiliation[HAU]{organization={Department of Mechanical Engineering, Huai'an University},
            city={Huai'an},
            state={Jiangsu},
            country={China}}"""
    new_front = """\\author{Sakar Mohammad Raziul Hasan Rifat, Akter Labani, Tao Zhang*, Saleh Mahamat Aboubakar OUSMANE, Boudjelkha Mohammed Djamel EDDINE}

Department of Mechanical Engineering, Huai'an University, Huai'an, Jiangsu, China

*Corresponding author. Email: taozhang2021@hau.edu.cn"""
    if old_front not in tex:
        sys.exit("Frontmatter block not found verbatim — main.tex header changed; update this script.")
    tex = tex.replace(old_front, new_front)

    old_kw = """\\begin{keyword}
acoustic anomaly detection \\sep machine condition monitoring \\sep convolutional neural network \\sep MIMII \\sep class imbalance \\sep edge computing
\\end{keyword}"""
    new_kw = "\\textbf{Keywords:} acoustic anomaly detection, machine condition monitoring, convolutional neural network, MIMII, class imbalance, edge computing"
    if old_kw not in tex:
        sys.exit("Keyword block not found verbatim — main.tex keywords changed; update this script.")
    tex = tex.replace(old_kw, new_kw)

    old_bib = "\\bibliographystyle{elsarticle-num}\n\\bibliography{references}"
    if old_bib not in tex:
        sys.exit("Bibliography commands not found verbatim — main.tex changed; update this script.")
    tex = tex.replace(old_bib, "\\section*{References}\n\n" + "\n\n".join(ref_lines))

    # \hyperlink/\hypertarget (used above for clickable citations and
    # cross-references) are hyperref macros; pandoc's LaTeX reader needs the
    # package declared to recognize them as links/bookmarks.
    old_docclass = "\\documentclass[review]{elsarticle}"
    if old_docclass not in tex:
        sys.exit("\\documentclass line not found verbatim — main.tex changed; update this script.")
    tex = tex.replace(old_docclass, old_docclass + "\n\\usepackage{hyperref}")

    with open(TEX_TMP, "w", encoding="utf-8") as f:
        f.write(tex)

    try:
        subprocess.run(
            ["pandoc", os.path.basename(TEX_TMP), "-o", DOCX_OUT, "--standalone"],
            cwd=PAPER_DIR, check=True,
        )
    finally:
        os.remove(TEX_TMP)

    # Pandoc's LaTeX reader silently drops raw \newpage — it isn't a
    # recognized macro — so page breaks before each top-level section have
    # to be set as a paragraph property directly on the generated docx.
    import docx
    d = docx.Document(DOCX_OUT)
    n_breaks = 0
    for p in d.paragraphs:
        if p.style.name == "Heading 1":
            p.paragraph_format.page_break_before = True
            n_breaks += 1
    d.save(DOCX_OUT)
    print(f"Set page-break-before on {n_breaks} top-level section headings.")

    print(f"Wrote {DOCX_OUT}")


if __name__ == "__main__":
    main()
