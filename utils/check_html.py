import argparse, sys, os, pathlib

# quick signatures for common non-HTML types
PDF_HDR = b"%PDF-"
ZIP_HDR = b"PK\x03\x04"  # docx/pptx/xlsx are ZIP containers
ELF_HDR = b"\x7fELF"
MZ_HDR  = b"MZ"

BLACKLIST_EXT = {
    ".pdf",".ppt",".pptx",".pps",".ppsx",".key",
    ".doc",".docx",".xls",".xlsx",".csv",".tsv",
    ".zip",".rar",".7z",".gz",".tgz",".bz2",".xz",".iso",".dmg",
    ".exe",".bin",".dll",".msi",".jar",
    ".png",".jpg",".jpeg",".gif",".bmp",".webp",".svg",".ico",
    ".mp3",".mp4",".m4v",".avi",".mov",".wmv",".webm",".ogg",".wav",
    ".css",".js",".json",".xml",".rss",".atom",
}

def is_html_bytes(b: bytes) -> bool:
    head = (b or b"")[:4096]
    if any(head.startswith(sig) for sig in (PDF_HDR, ZIP_HDR, ELF_HDR, MZ_HDR)):
        return False
    # heuristic: look for doctype or <html tag in first 4KB
    try:
        text = head.decode("utf-8", errors="ignore").lower()
    except Exception:
        return False
    return ("<!doctype html" in text) or ("<html" in text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages_dir", default="data/pages", help="folder with saved HTML files")
    ap.add_argument("--report", default="data/report/html_check.txt")
    args = ap.parse_args()

    pages = pathlib.Path(args.pages_dir)
    bad = []
    total = 0
    for p in pages.rglob("*"):
        if not p.is_file(): 
            continue
        total += 1
        ext = p.suffix.lower()
        if ext in BLACKLIST_EXT:
            bad.append((str(p), f"blacklisted extension {ext}"))
            continue
        try:
            b = p.read_bytes()
        except Exception as e:
            bad.append((str(p), f"read error: {e}"))
            continue
        if p.suffix.lower() in {".html", ".htm"}:
            continue
        if not is_html_bytes(b):
            bad.append((str(p), "content not recognized as HTML"))
    os.makedirs(os.path.dirname(args.report), exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as f:
        f.write(f"Checked files: {total}\n")
        f.write(f"Non-HTML or blacklisted: {len(bad)}\n")
        for path, reason in bad:
            f.write(f"{reason} :: {path}\n")
    print(f"Checked {total} files, flagged {len(bad)}. Report: {args.report}")
    sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()