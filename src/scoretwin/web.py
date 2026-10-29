"""Local browser interface for the ScoreTwin prototype."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .imaging import load_grayscale
from .pipeline import analyze_array, render_overlay
from .musicxml import build_musicxml
from .proposal import get_default_model, load_model

app = FastAPI(title="ScoreTwin / IDSR", version="0.1.0")
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}
MODEL_PATH = os.environ.get("SCORETWIN_MODEL")
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScoreTwin / IDSR</title>
<style>
:root{color-scheme:dark;font-family:Inter,system-ui,sans-serif;background:#10131a;color:#edf0f6}
body{max-width:1050px;margin:0 auto;padding:28px 18px}
h1{letter-spacing:-.04em;margin-bottom:5px}.muted{color:#9da8b9}.panel{border:1px solid #303949;border-radius:14px;padding:18px;margin:18px 0;background:#171c26}
button{background:#b7f36b;color:#151a12;font-weight:700;border:0;border-radius:9px;padding:12px 18px;cursor:pointer}
input{margin:12px 0}img{width:100%;height:auto;border-radius:8px;border:1px solid #303949}
pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#ced7e5}a{color:#b7f36b}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}@media(max-width:700px){.grid{grid-template-columns:1fr}}
.status{padding:10px 0;font-weight:600}
</style>
</head>
<body>
<h1>ScoreTwin <span class="muted">/ IDSR</span></h1>
<p class="muted">Geometry-first score digitization prototype. The source file is preserved exactly; symbolic proposals remain provisional.</p>
<div class="panel">
<form id="upload"><label for="file">Choose a score image or PDF</label><br><input id="file" name="file" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.pdf" required><br><button id="submit" type="submit">Analyze page</button></form>
<div id="status" class="status" role="status"></div>
</div>
<div id="results" hidden>
<div class="grid">
<section class="panel"><h2>Diagnostic overlay</h2><img id="overlay" alt="Detected staff and notehead proposals"></section>
<section class="panel"><h2>Scene summary</h2><pre id="summary"></pre><a id="download" download="provisional.musicxml">Download provisional MusicXML</a></section>
</div>
<section class="panel"><h2>Uncertainty and warnings</h2><pre id="warnings"></pre></section>
</div>
<script>
const form=document.getElementById('upload');
form.addEventListener('submit',async(event)=>{
 event.preventDefault();
 const file=document.getElementById('file').files[0];
 if(!file)return;
 const button=document.getElementById('submit');button.disabled=true;
 const status=document.getElementById('status');status.textContent='Processing locally…';
 try{
  const body=new FormData();body.append('file',file);
  const response=await fetch('/api/analyze',{method:'POST',body});
  const result=await response.json();
  if(!response.ok)throw new Error(result.detail||'Analysis failed');
  document.getElementById('overlay').src='data:image/png;base64,'+result.overlay_png_base64;
  document.getElementById('summary').textContent=JSON.stringify(result.summary,null,2);
  document.getElementById('warnings').textContent=result.warnings.join('\n');
  const blob=new Blob([result.musicxml],{type:'application/vnd.recordare.musicxml+xml'});
  const url=URL.createObjectURL(blob);document.getElementById('download').href=url;
  document.getElementById('results').hidden=false;
  status.textContent='Analysis finished. Review every pitch and all warnings before using the MusicXML.';
 }catch(error){status.textContent=error.message||String(error);}
 finally{button.disabled=false;}
});
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return PAGE


@app.post("/api/analyze")
async def analyze(file: UploadFile = File(...)) -> dict:
    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail="Upload a PNG, JPEG, TIFF, BMP or PDF file.")
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    if len(contents) > 40 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File size limit is 40 MiB.")
    try:
        with tempfile.TemporaryDirectory(prefix="scoretwin-") as directory:
            input_path = Path(directory) / ("source" + suffix)
            input_path.write_bytes(contents)
            gray = load_grayscale(input_path)
            model = load_model(MODEL_PATH) if MODEL_PATH and Path(MODEL_PATH).exists() else get_default_model()
            scene = analyze_array(gray, model=model)
            overlay = render_overlay(gray, scene)
            success, encoded = cv2.imencode(".png", overlay)
            if not success:
                raise RuntimeError("Overlay encoding failed.")
            xml = build_musicxml(scene)
            return {
                "summary": {
                    "filename": filename,
                    "width": scene["image"]["width"],
                    "height": scene["image"]["height"],
                    "staff_count": len(scene["staffs"]),
                    "proposals": scene["proposal_count"],
                    "notehead_proposals": scene["note_count"],
                    "runtime_seconds": scene["runtime_seconds"],
                },
                "warnings": scene["warnings"],
                "scene": scene,
                "musicxml": xml,
                "overlay_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
            }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
