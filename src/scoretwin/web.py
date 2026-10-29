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
from .pipeline import analyze_array, render_geometry_scene
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
canvas{width:100%;height:380px;background:#0d1118;border-radius:10px;border:1px solid #303949;touch-action:none}
#pageCanvas{height:auto;display:block;background:#f4f4f4;cursor:crosshair}
#scene3d{height:380px}.status{padding:10px 0;font-weight:600}
@media(max-width:700px){.grid{grid-template-columns:1fr}canvas{height:300px}#pageCanvas{height:auto}#scene3d{height:300px}}
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
<section class="panel"><h2>2D geometry editor</h2><p class="muted">Drag a proposal box to adjust its geometry. This edits the reconstruction proposal, not the preserved source file.</p><canvas id="pageCanvas" width="1100" height="400" aria-label="Draggable notehead proposals over the source page"></canvas></section>
<section class="panel"><h2>Scene summary</h2><pre id="summary"></pre><div><a id="download" download="provisional.musicxml">Download provisional MusicXML</a></div><button id="export-scene" type="button">Download edited scene JSON</button><button id="reset-edits" type="button">Reset edits</button></section>
</div>
<section class="panel"><h2>Structural 2D render</h2>
<p class="muted">Only currently modeled staff lines, barlines and notehead proposals are drawn. This is not yet a pixel-identical replica; the source raster remains the identity anchor.</p>
<img id="geometry-render" alt="Structural render of currently modeled primitives">
<button id="render-current" type="button">Render current geometry</button>
</section>
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
let currentScene=null, originalScene=null, yaw=-0.38, tilt=-0.2, dragging=false, draggingNote=-1, lastX=0, lastY=0;
const canvas=document.getElementById('scene3d'),ctx=canvas.getContext('2d');
const pageCanvas=document.getElementById('pageCanvas'),pageCtx=pageCanvas.getContext('2d');
const sourcePreview=new Image();
sourcePreview.onload=()=>{pageCanvas.width=sourcePreview.naturalWidth;pageCanvas.height=sourcePreview.naturalHeight;drawPage();};
function pagePoint(event){
 const rect=pageCanvas.getBoundingClientRect();
 return [(event.clientX-rect.left)/rect.width*currentScene.image.width,
         (event.clientY-rect.top)/rect.height*currentScene.image.height];
}
function syncPrimitive(note){
 const primitive=(currentScene.primitives||[]).find(item=>item.id===(note.primitive_id||note.id));
 if(!primitive)return;
 const box=note.bbox;
 primitive.geometry.center_x_px=Number(note.center[0]);
 primitive.geometry.center_y_px=Number(note.center[1]);
 primitive.geometry.radius_x_px=Number(box[2])/2;
 primitive.geometry.radius_y_px=Number(box[3])/2;
 primitive.geometry.bbox_px=box.slice();
 primitive.symbolic=primitive.symbolic||{};
 primitive.symbolic.pitch_proposal={...note.pitch};
}
function updatePitchFromGeometry(note){
 const staff=currentScene.staffs[note.staff_index];if(!staff)return;
 const continuous=(staff.bottom-note.center[1])/(staff.spacing/2);
 const step=Math.round(continuous);
 const letters=['C','D','E','F','G','A','B'],absoluteIndex=2+step;
 const octaveDelta=Math.floor(absoluteIndex/7),letterIndex=((absoluteIndex%7)+7)%7;
 note.diatonic_step_from_bottom_line=step;
 note.continuous_staff_step=continuous;
 note.staff_position_offset_steps=continuous-step;
 if(!note.manually_edited_pitch)note.pitch={step:letters[letterIndex],octave:4+octaveDelta};
 syncPrimitive(note);
}
function drawPage(){
 if(!currentScene||!sourcePreview.complete)return;
 pageCtx.clearRect(0,0,pageCanvas.width,pageCanvas.height);
 pageCtx.drawImage(sourcePreview,0,0,pageCanvas.width,pageCanvas.height);
 const sx=pageCanvas.width/currentScene.image.width,sy=pageCanvas.height/currentScene.image.height;
 currentScene.notes.forEach((note,index)=>{
  const box=note.bbox,x=box[0]*sx,y=box[1]*sy,w=box[2]*sx,h=box[3]*sy;
  pageCtx.strokeStyle=index===draggingNote?'#b7f36b':'#e56f79';pageCtx.lineWidth=Math.max(1.4,pageCanvas.width/900);
  pageCtx.strokeRect(x,y,w,h);
  pageCtx.fillStyle='#e56f79';pageCtx.font=(Math.max(10,pageCanvas.width/100))+'px system-ui';
  pageCtx.fillText(note.pitch.step+note.pitch.octave+'?',x,y-3<10?y+h+12:y-3);
 });
 currentScene.unknown_objects.forEach(item=>{
  const x=item.bbox[0]*sx,y=item.bbox[1]*sy,w=item.bbox[2]*sx,h=item.bbox[3]*sy;
  pageCtx.strokeStyle='#e3b35b';pageCtx.lineWidth=1;pageCtx.strokeRect(x,y,w,h);
 });
}
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
   currentScene.notes[index].manually_edited_pitch=true;
   syncPrimitive(currentScene.notes[index]);
   draw3d();drawPage();
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
  currentScene.source=result.source;
  originalScene=JSON.parse(JSON.stringify(currentScene));
  sourcePreview.src='data:image/png;base64,'+result.source_preview_png_base64;
  document.getElementById('summary').textContent=JSON.stringify(result.summary,null,2);
  document.getElementById('warnings').textContent=result.warnings.join('\n');
  document.getElementById('geometry-render').src='data:image/png;base64,'+result.geometry_render_png_base64;
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
document.getElementById('render-current').addEventListener('click',async()=>{
 if(!currentScene)return;
 const button=document.getElementById('render-current');button.disabled=true;
 try{
  const response=await fetch('/api/render',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(currentScene)});
  const result=await response.json();if(!response.ok)throw new Error(result.detail||'Render failed');
  document.getElementById('geometry-render').src='data:image/png;base64,'+result.geometry_render_png_base64;
  document.getElementById('status').textContent='Structural render updated from the current geometry proposals.';
 }catch(error){document.getElementById('status').textContent=error.message||String(error);}
 finally{button.disabled=false;}
});
document.getElementById('export-scene').addEventListener('click',()=>{
 if(!currentScene)return;
 const blob=new Blob([JSON.stringify(currentScene,null,2)],{type:'application/json'});
 const url=URL.createObjectURL(blob),anchor=document.createElement('a');
 anchor.href=url;anchor.download='scoretwin-edited-scene.json';anchor.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.getElementById('reset-edits').addEventListener('click',()=>{
 if(!originalScene)return;
 currentScene=JSON.parse(JSON.stringify(originalScene));
 buildEditor();draw3d();drawPage();
 document.getElementById('status').textContent='Edits reset to the initial proposals. The source file was never modified.';
});
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
pageCanvas.addEventListener('pointerdown',event=>{
 if(!currentScene)return;
 const point=pagePoint(event);
 for(let i=currentScene.notes.length-1;i>=0;i--){
  const note=currentScene.notes[i],b=note.bbox,pad=currentScene.staffs[note.staff_index]?.spacing*0.35||4;
  if(point[0]>=b[0]-pad&&point[0]<=b[0]+b[2]+pad&&point[1]>=b[1]-pad&&point[1]<=b[1]+b[3]+pad){
   draggingNote=i;pageCanvas.setPointerCapture(event.pointerId);drawPage();return;
  }
 }
});
pageCanvas.addEventListener('pointermove',event=>{
 if(draggingNote<0||!currentScene)return;
 const point=pagePoint(event),note=currentScene.notes[draggingNote];
 note.center=[Math.max(0,Math.min(currentScene.image.width-1,point[0])),Math.max(0,Math.min(currentScene.image.height-1,point[1]))];
 const b=note.bbox;note.bbox=[Math.round(note.center[0]-b[2]/2),Math.round(note.center[1]-b[3]/2),b[2],b[3]];
 note.manually_edited_geometry=true;updatePitchFromGeometry(note);drawPage();draw3d();
});
pageCanvas.addEventListener('pointerup',()=>{
 if(draggingNote>=0){draggingNote=-1;buildEditor();drawPage();document.getElementById('status').textContent='Geometry proposal updated. The original source hash and bytes are unchanged.';}
});
pageCanvas.addEventListener('pointercancel',()=>{draggingNote=-1;drawPage();});
window.addEventListener('resize',()=>{draw3d();drawPage();});
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
            source_sha256 = hashlib.sha256(contents).hexdigest()
            scene["source"] = {
                "filename": filename,
                "sha256": source_sha256,
                "bytes": len(contents),
                "page_index": 0 if suffix == ".pdf" else None,
                "server_persisted": False,
            }
            preview = gray
            preview_scale = min(1.0, 1600.0 / max(gray.shape[1], gray.shape[0]))
            if preview_scale < 1.0:
                preview = cv2.resize(
                    gray,
                    (max(1, int(round(gray.shape[1] * preview_scale))),
                     max(1, int(round(gray.shape[0] * preview_scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            success, encoded = cv2.imencode(".png", preview)
            if not success:
                raise RuntimeError("Preview encoding failed.")
            xml = build_musicxml(scene)
            return {
                "summary": {
                    "filename": filename,
                    "source_sha256": source_sha256,
                    "width": scene["image"]["width"],
                    "height": scene["image"]["height"],
                    "staff_count": len(scene["staffs"]),
                    "proposals": scene["proposal_count"],
                    "notehead_proposals": scene["note_count"],
                    "runtime_seconds": scene["runtime_seconds"],
                },
                "warnings": scene["warnings"],
                "source": scene["source"],
                "scene": scene,
                "musicxml": xml,
                "source_preview_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "geometry_render_png_base64": _render_base64(gray.shape, scene),
            }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _render_base64(shape: tuple[int, int], scene: dict) -> str:
    rendered = render_geometry_scene(shape, scene)
    success, encoded = cv2.imencode(".png", rendered)
    if not success:
        raise RuntimeError("Could not encode geometry render.")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


@app.post("/api/render")
def render_current_geometry(scene: dict = Body(...)) -> dict:
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise HTTPException(status_code=400, detail="Unrecognized ScoreTwin scene schema.")
    try:
        height = int(scene["image"]["height"])
        width = int(scene["image"]["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Scene image dimensions are required.") from exc
    if height < 32 or width < 32 or height * width > 80_000_000:
        raise HTTPException(status_code=413, detail="Image dimensions are invalid or exceed the 80 megapixel limit.")
    try:
        return {"geometry_render_png_base64": _render_base64((height, width), scene)}
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
