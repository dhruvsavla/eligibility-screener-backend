"""
Diagnoses where PDF text is lost. Run: python diagnose_pdf.py <path_to_pdf>
"""
import sys
import pdfplumber

def diagnose(pdf_path: str):
    print("=" * 70)
    print("PDF EXTRACTION DIAGNOSTIC")
    print("=" * 70)

    with pdfplumber.open(pdf_path) as pdf:
        n_pages = len(pdf.pages)
        full_text = "\n".join(page.extract_text() or "" for page in pdf.pages)

    print(f"\n[1] PDF has {n_pages} pages")
    print(f"[1] Full extracted text length: {len(full_text):,} chars")

    lower = full_text.lower()
    markers = {
        "inclusion criteria": lower.find("inclusion criteria"),
        "exclusion criteria": lower.find("exclusion criteria"),
        "study population":   lower.find("study population"),
        "8.1":                lower.find("8.1"),
        "8.2":                lower.find("8.2"),
    }
    print(f"\n[2] Eligibility marker positions in full text:")
    for marker, pos in markers.items():
        if pos == -1:
            print(f"    '{marker}': NOT FOUND")
        else:
            pct = (pos / len(full_text)) * 100
            print(f"    '{marker}': char {pos:,} ({pct:.0f}% through document)")

    print(f"\n[3] What common truncation limits would capture:")
    for limit in [8000, 12000, 15000, 50000]:
        has_incl = "inclusion criteria" in full_text[:limit].lower()
        has_excl = "exclusion criteria" in full_text[:limit].lower()
        print(f"    text[:{limit}]: inclusion={'✓' if has_incl else '✗'} "
              f"exclusion={'✓' if has_excl else '✗'}")

    excl_pos = lower.find("exclusion criteria")
    if excl_pos != -1:
        print(f"\n[4] Exclusion section IS present in full text. Preview:")
        print("    " + full_text[excl_pos:excl_pos+400].replace("\n", "\n    "))
    else:
        print("\n[4] WARNING: 'exclusion criteria' not found in full extracted text!")

if __name__ == "__main__":
    diagnose(sys.argv[1])
