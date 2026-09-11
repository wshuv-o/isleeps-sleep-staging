"""Neutralise the pdfTeX-only Pantone spot-colour block for a local preview build.

PREVIEW ONLY. The IEEE Access class defines its blue through spot-colour
primitives that exist in pdfTeX but not XeTeX, which is what tectonic runs. The
repository's class file is never touched: this patches a copy in the scratchpad,
and the real submission still builds on Overleaf with pdfLaTeX and the true
Pantone 3015 C.
"""
import io
import os

SP = (r"C:\Users\ESMEAB~1\AppData\Local\Temp\claude"
      r"\d--proc-isleeps-sleep-staging\44090db4-9abc-4c5e-8342-43ef69cafb9a"
      r"\scratchpad\preview")
CLS = os.path.join(SP, "ieeeaccess.cls")

DROP = (r"\NewSpotColorSpace", r"\AddSpotColor", r"\SetPageColorSpace")
SPOT_BLUE = r"\definecolor{accessblue}{spotcolor}"
# closest CMYK equivalent of PANTONE 3015 C, from the class's own 1 0.3 0 0.2
CMYK_BLUE = r"  \definecolor{accessblue}{cmyk}{1,0.3,0,0.2}%"

lines = io.open(CLS, encoding="utf-8", errors="ignore").read().split("\n")
out, n = [], 0
for ln in lines:
    st = ln.strip()
    if any(st.startswith(t) for t in DROP):
        out.append("% [preview] " + ln)
        n += 1
    elif st.startswith(SPOT_BLUE):
        out.append("% [preview] " + ln)
        out.append(CMYK_BLUE)
        n += 1
    else:
        out.append(ln)

io.open(CLS, "w", encoding="utf-8").write("\n".join(out))
print("commented %d spot-colour lines; accessblue now CMYK" % n)
