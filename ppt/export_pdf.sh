#!/usr/bin/env bash
# Export ppt/G28_presentation.pptx to ppt/G28_presentation.pdf with Microsoft PowerPoint (WSL -> Windows).
# PowerPoint is driven through COM from PowerShell; the deck is copied to the Windows temp folder first
# because PowerPoint does not reliably open files on \\wsl.localhost paths.
#   ppt/export_pdf.sh
set -euo pipefail
cd "$(dirname "$0")"
SRC=G28_presentation.pptx
OUT=G28_presentation.pdf
WTEMP=$(powershell.exe -NoProfile -Command '[Console]::Write($env:TEMP)')
LTEMP=$(wslpath -u "$WTEMP")
cp "$SRC" "$LTEMP/g28_export.pptx"
rm -f "$LTEMP/g28_export.pdf"
powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "
\$ErrorActionPreference = 'Stop'
\$app = New-Object -ComObject PowerPoint.Application
\$pres = \$app.Presentations.Open('$WTEMP\\g28_export.pptx', -1, 0, 0)   # ReadOnly, not untitled, no window
\$pres.SaveAs('$WTEMP\\g28_export.pdf', 32)                             # 32 = ppSaveAsPDF
\$pres.Close()
if (\$app.Presentations.Count -eq 0) { \$app.Quit() }                     # never close the user's own decks
"
cp "$LTEMP/g28_export.pdf" "$OUT" && chmod 644 "$OUT"
rm -f "$LTEMP/g28_export.pptx" "$LTEMP/g28_export.pdf"
echo "wrote ppt/$OUT"
