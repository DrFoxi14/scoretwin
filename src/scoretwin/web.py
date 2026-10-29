"""Local browser interface for the ScoreTwin prototype."""

from __future__ import annotations

import base64
import hashlib
import os
import tempfile
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
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
body{max-width:1150px;margin:0 auto;padding:28px 18px}
h1{letter-spacing:-.04em;margin-bottom:5px}.muted{color:#9da8b9}.panel{border:1px solid #303949;border-radius:14px;padding:18px;margin:18px 0;background:#171c26}
button{background:#b7f36b;color:#151a12;font-weight:700;border:0;border-radius:9px;padding:11px 16px;cursor:pointer;margin:4px}
input{margin:12px 0}img{width:100%;height:auto;border-radius:8px;border:1px solid #303949}
pre{white-space:pre-wrap;overflow-wrap:anywhere;color:#ced7e5}a{color:#b7f36b}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.editor-row{display:flex;gap:12px;align-items:center;justify-content:space-between;border-top:1px solid #303949;padding:10px 0}.editor-row select{background:#10131a;color:#edf0f6;padding:7px;border:1px solid #46536a;border-radius:7px}
canvas{width:100%;height:380px;background:#0d1118;border-radius:10px;border:1px solid #303949;touch-action:none}.status{padding:10px 0;font-weight:600}
@media(max-width:700px){.grid{grid-template-columns:1fr}canvas{height:300px}}
</style>
</head>
<body>
<h1>ScoreTwin <span class="muted">/ IDSR</span></h1>
<p class="muted">Geometry-first score digitization prototype. The source stays the identity anchor; symbolic proposals remain provisional.</p>
<div class="panel">
<form id="upload"><label for="file">Choose a score image or PDF</label><br><input id="file" name="file" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.pdf" required><br><button id="submit" type="submit">Analyze page</button></form>
<div id="status" class="status" role="status"></div>
</div>
<div id="results" hidden>
<div class="grid">
<section class="panel"><h2>2D overlay</h2><img id="overlay" alt="Detected staff and notehead proposals"></section>
<section class="panel"><h2>Scene summary</h2><pre id="summary"></pre><a id="download" download="provisional.musicxml">Download provisional MusicXML</a></section>
</div>
<section class="panel"><h2>3D semantic layer view</h2>
<p class="muted">Drag to rotate. The layer offset is a visualization convention, not measured physical ink depth.</p>
<canvas id="scene3d" width="1100" height="400" aria-label="Interactive semantic layer preview"></canvas><br>
<label for="lambda">Layer separation λ</label> <input id="lambda" type="range" min="0" max="1" step="0.02" value="0.55">
<button id="reset-view" type="button">Reset view</button>
</section>
<section class="panel"><h2>Review and edit pitch proposals</h2>
<p class="muted">Changing a pitch changes the symbolic proposal only. The original image and its recorded hash do not change. Durations, key signatures, accidentals, voices and part assignment still require review.</p>
<div id="editor"></div><button id="export-edits" type="button">Regenerate MusicXML from edits</button>
</section>
<section class="panel"><h2>Warnings</h2><pre id="warnings"></pre></section>
</div>
<script>
let currentScene=null, yaw=-0.38, tilt=-0.2, dragging=false, lastX=0, lastY=0;
const canvas=document.getElementById('scene3d'),ctx=canvas.getContext('2d');
function rotatePoint(x,y,z){
 const cy=Math.cos(yaw),sy=Math.sin(yaw),cx=Math.cos(tilt),sx=Math.sin(tilt);
 const xx=x*cy-z*sy,zz=x*sy+z*cy;
 return [xx,y*cx-zz*sx,y*sx+zz*cx];
}
function projectPoint(x,y,z){
 const width=currentScene.image.width,height=currentScene.image.height;
 const p=rotatePoint(x-width/2,y-height/2,z);
 const scale=Math.min((canvas.width-90)/Math.max(width,1),(canvas.height-70)/Math.max(height,1));
 return [canvas.width/2+p[0]*scale,canvas.height/2+p[1]*scale,p[2]];
}
function draw3d(){
 ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.fillStyle='#0d1118';ctx.fillRect(0,0,canvas.width,canvas.height);
 if(!currentScene)return;
 const width=currentScene.image.width,height=currentScene.image.height;
 const sep=Number(document.getElementById('lambda').value)*Math.max(16,currentScene.staffs.length*10);
 const cx0=width/2,cy0=height/2;
 const lines=[];
 currentScene.staffs.forEach((staff,si)=>{
  staff.lines.forEach(y=>lines.push([[0-cx0,y-cy0,0],[width-cx0,y-cy0,0]]));
 });
 ctx.lineWidth=1.2;ctx.strokeStyle='#4a9a77';
 lines.forEach(pair=>{
  const a=projectPoint(pair[0][0]+cx0,pair[0][1]+cy0,pair[0][2]);
  const b=projectPoint(pair[1][0]+cx0,pair[1][1]+cy0,pair[1][2]);
  ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
 });
 currentScene.notes.forEach((note,index)=>{
  const x=Number(note.center[0])-cx0,y=Number(note.center[1])-cy0;
  const z=sep*(0.65+(note.staff_index%3)*0.16);
  const p=projectPoint(x+cx0,y+cy0,z);
  const r=Math.max(3,Math.min(8,Number(note.bbox[2])*0.28));
  ctx.fillStyle='#e56f79';ctx.beginPath();ctx.ellipse(p[0],p[1],r*1.2,r, yaw*0.15,0,Math.PI*2);ctx.fill();
  ctx.fillStyle='#edf0f6';ctx.font='11px system-ui';ctx.fillText(note.pitch.step+note.pitch.octave+'?',p[0]+r+2,p[1]-r);
 });
 currentScene.unknown_objects.forEach(item=>{
  const x=Number(item.center[0])-cx0,y=Number(item.center[1])-cy0;
  const p=projectPoint(x+cx0,y+cy0,sep*0.3);
  ctx.fillStyle='#e3b35b';ctx.fillRect(p[0]-3,p[1]-3,6,6);
 });
 ctx.fillStyle='#9da8b9';ctx.font='12px system-ui';
 ctx.fillText('x/y: observed page coordinates · z: semantic visualization only',16,24);
}
function buildEditor(){
 const holder=document.getElementById('editor');holder.replaceChildren();
 if(!currentScene||!currentScene.notes.length){holder.textContent='No notehead proposals are ready for pitch editing.';return;}
 const letters=['C','D','E','F','G','A','B'];
 currentScene.notes.forEach((note,index)=>{
  const row=document.createElement('div');row.className='editor-row';
  const title=document.createElement('span');title.textContent='Proposal '+(index+1)+' · '+note.id+' · '+Math.round(note.center[0])+','+Math.round(note.center[1]);
  const select=document.createElement('select');select.setAttribute('aria-label','Pitch for '+note.id);
  for(let octave=2;octave<=7;octave++)letters.forEach(letter=>{
   const option=document.createElement('option');option.value=letter+octave;option.textContent=letter+octave;
   if(letter===note.pitch.step&&octave===Number(note.pitch.octave))option.selected=true;
   select.appendChild(option);
  });
  select.addEventListener('change',()=>{
   const match=select.value.match(/^([A-G])([2-7])$/);if(!match)return;
   currentScene.notes[index].pitch={step:match[1],octave:Number(match[2])};
   currentScene.notes[index].uncertainty=currentScene.notes[index].uncertainty||[];
   currentScene.notes[index].manually_edited=true;
   draw3d();
  });
  row.append(title,select);holder.appendChild(row);
 });
}
const form=document.getElementById('upload');
form.addEventListener('submit',async(event)=>{
 event.preventDefault();const file=document.getElementById('file').files[0];if(!file)return;
 const button=document.getElementById('submit');button.disabled=true;const status=document.getElementById('status');status.textContent='Processing locally…';
 try{
  const body=new FormData();body.append('file',file);
  const response=await fetch('/api/analyze',{method:'POST',body});const result=await response.json();
  if(!response.ok)throw new Error(result.detail||'Analysis failed');
  currentScene=result.scene;
  document.getElementById('overlay').src='data:image/png;base64,'+result.overlay_png_base64;
  document.getElementById('summary').textContent=JSON.stringify(result.summary,null,2);
  document.getElementById('warnings').textContent=result.warnings.join('\n');
  setDownload(result.musicxml);document.getElementById('results').hidden=false;
  buildEditor();draw3d();
  status.textContent='Analysis finished. Review every pitch and all warnings before using the MusicXML.';
 }catch(error){status.textContent=error.message||String(error);}
 finally{button.disabled=false;}
});
function setDownload(xml){
 const url=URL.createObjectURL(new Blob([xml],{type:'application/vnd.recordare.musicxml+xml'}));
 const anchor=document.getElementById('download');if(anchor.dataset.url)URL.revokeObjectURL(anchor.dataset.url);
 anchor.href=url;anchor.dataset.url=url;
}
document.getElementById('lambda').addEventListener('input',draw3d);
document.getElementById('reset-view').addEventListener('click',()=>{yaw=-0.38;tilt=-0.2;draw3d();});
canvas.addEventListener('pointerdown',event=>{dragging=true;lastX=event.clientX;lastY=event.clientY;canvas.setPointerCapture(event.pointerId);});
canvas.addEventListener('pointermove',event=>{if(!dragging)return;yaw+=(event.clientX-lastX)*0.008;tilt+=(event.clientY-lastY)*0.008;tilt=Math.max(-1.2,Math.min(1.2,tilt));lastX=event.clientX;lastY=event.clientY;draw3d();});
canvas.addEventListener('pointerup',()=>{dragging=false;});canvas.addEventListener('pointercancel',()=>{dragging=false;});
document.getElementById('export-edits').addEventListener('click',async()=>{
 if(!currentScene)return;
 const button=document.getElementById('export-edits');button.disabled=true;
 try{
  const response=await fetch('/api/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentScene)});
  const result=await response.json();if(!response.ok)throw new Error(result.detail||'Export failed');
  setDownload(result.musicxml);document.getElementById('status').textContent='MusicXML regenerated. Pitch edits are included; remaining ambiguities are still marked.';
 }catch(error){document.getElementById('status').textContent=error.message||String(error);}
 finally{button.disabled=false;}
});
window.addEventListener('resize',draw3d);
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
                    "source_sha256": hashlib.sha256(contents).hexdigest(),
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


@app.post("/api/export")
def export_edits(scene: dict = Body(...)) -> dict:
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise HTTPException(status_code=400, detail="Unrecognized ScoreTwin scene schema.")
    if len(scene.get("notes", [])) > 5000:
        raise HTTPException(status_code=413, detail="Too many note proposals in one scene.")
    for note in scene.get("notes", []):
        pitch = note.get("pitch", {})
        if pitch.get("step") not in {"A", "B", "C", "D", "E", "F", "G"}:
            raise HTTPException(status_code=422, detail="Pitch step must be a natural note A-G.")
        try:
            octave = int(pitch.get("octave"))
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=422, detail="Pitch octave must be an integer.") from exc
        if octave < 0 or octave > 9:
            raise HTTPException(status_code=422, detail="Pitch octave must be between 0 and 9.")
    return {"musicxml": build_musicxml(scene)}
