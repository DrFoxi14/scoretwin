"""Local browser interface for the ScoreTwin prototype."""

from __future__ import annotations

import base64
import json
import os
import tempfile
from pathlib import Path

import cv2
from fastapi import Body, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

from .gltf_export import build_gltf, validate_embedded_gltf
from .imaging import load_grayscale
from .pipeline import (
    compile_digital_score_audio,
    compile_digital_score_pdf,
    process_file,
    render_geometry_scene,
)
from .musicxml import build_musicxml

app = FastAPI(title="ScoreTwin / IDSR", version="0.1.0")
ALLOWED_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".pdf"}
MODEL_PATH = os.environ.get("SCORETWIN_MODEL")
PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ScoreTwin / IDSR</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
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
#cornerCanvas{height:auto;display:block;background:#f4f4f4;cursor:crosshair}
#scene3d{height:380px}.status{padding:10px 0;font-weight:600}
@media(max-width:700px){.grid{grid-template-columns:1fr}canvas{height:300px}#pageCanvas{height:auto}#scene3d{height:300px}}
</style>
</head>
<body>
<h1>ScoreTwin <span class="muted">/ IDSR</span></h1>
<p class="muted">Geometry-first score digitization prototype. The source stays the identity anchor; symbolic proposals remain provisional.</p>
<div class="panel">
<form id="upload"><label for="file">Choose an image or PDF</label><br><input id="file" name="file" type="file" accept=".png,.jpg,.jpeg,.tif,.tiff,.bmp,.pdf"><br><label for="camera-file">Or take a photo with the phone camera</label><br><input id="camera-file" name="camera-file" type="file" accept="image/*" capture="environment"><br><label for="page">PDF page index (0-based; ignored for images)</label><br><input id="page" name="page" type="number" min="0" max="10000" value="0"><br><label style="display:inline-flex;align-items:center;gap:6px;cursor:pointer"><input id="auto-substrate" name="auto-substrate" type="checkbox" checked> Auto-crop paper on desk / background (homography H)</label><br><button id="submit" type="submit">Analyze page</button></form>
<div id="corner-panel" hidden><h2>Optional perspective correction</h2><p class="muted">On a photo, tap the page corners in order: top left, top right, bottom right, bottom left. This assumes a flat page.</p><button id="mark-corners" type="button">Select four page corners</button><button id="clear-corners" type="button">Clear corners</button><p id="corner-status" class="muted" aria-live="polite"></p><canvas id="cornerCanvas" aria-label="Preview for selecting page corners"></canvas></div>
<div id="status" class="status" role="status"></div>
</div>
<div id="results" hidden>
<div class="grid">
<section class="panel"><h2>2D geometry & notation editor</h2><p class="muted">Drag a proposal box to adjust its geometry. Layers are displayed without modifying the preserved source file.</p>
<div style="display:flex;gap:14px;flex-wrap:wrap;margin:10px 0;font-size:13px;align-items:center;background:#10131a;padding:8px 12px;border-radius:8px;border:1px solid #303949">
  <span style="font-weight:600;color:#9da8b9">Straturi vizuale:</span>
  <label style="cursor:pointer;display:inline-flex;align-items:center;gap:5px"><input id="show-notes" type="checkbox" checked> <span style="color:#e56f79">■</span> Note & Înălțimi</label>
  <label style="cursor:pointer;display:inline-flex;align-items:center;gap:5px"><input id="show-lyrics" type="checkbox" checked> <span style="color:#34d399">■</span> Versuri / Text (Română)</label>
  <label style="cursor:pointer;display:inline-flex;align-items:center;gap:5px"><input id="show-satb" type="checkbox" checked> <span style="color:#a78bfa">■</span> Voci SATB & Chei</label>
  <label style="cursor:pointer;display:inline-flex;align-items:center;gap:5px"><input id="show-fingers" type="checkbox" checked> <span style="color:#38bdf8">■</span> Ocluzii Degete</label>
</div>
<canvas id="pageCanvas" width="1100" height="400" aria-label="Draggable notehead proposals over the source page"></canvas></section>
<section class="panel"><h2>Scene summary & Playback</h2><pre id="summary"></pre>
<div style="margin: 12px 0; background: #161b22; border: 1px solid #303949; border-radius: 8px; padding: 12px;">
  <div style="font-weight: 600; margin-bottom: 8px; color: #58a6ff; display: flex; align-items: center; gap: 8px;">
    <span>🎧 Ascultă redarea digitalizată (MuseScore 4 Engine):</span>
  </div>
  <audio id="score-audio-player" controls style="width: 100%; height: 36px; margin-bottom: 10px;" hidden></audio>
  <div style="display: flex; gap: 8px; flex-wrap: wrap;">
    <a id="download-digital-pdf" download="score_digital.pdf" style="background:#238636;color:#ffffff;text-decoration:none;padding:8px 14px;border-radius:6px;font-weight:600;display:inline-flex;align-items:center;gap:6px;cursor:pointer;">📄 Download Digitalized Score (PDF)</a>
    <a id="download-audio-mp3" download="score_audio.mp3" style="background:#8957e5;color:#ffffff;text-decoration:none;padding:8px 14px;border-radius:6px;font-weight:600;display:inline-flex;align-items:center;gap:6px;cursor:pointer;">🎧 Download Audio (MP3)</a>
    <a id="download-midi" download="score_audio.mid" style="background:#da3633;color:#ffffff;text-decoration:none;padding:8px 14px;border-radius:6px;font-weight:600;display:inline-flex;align-items:center;gap:6px;cursor:pointer;">🎵 Download MIDI</a>
    <a id="download" download="provisional.musicxml" style="background:#1f6feb;color:#ffffff;text-decoration:none;padding:8px 14px;border-radius:6px;font-weight:600;display:inline-flex;align-items:center;gap:6px;cursor:pointer;">🎼 Download MusicXML</a>
  </div>
</div>
<button id="export-scene" type="button">Download edited scene JSON</button><button id="reset-edits" type="button">Reset edits</button></section>
</div>
<section class="panel" id="satb-panel" hidden>
<h2>Ansamblu Coral SATB & Strat Versuri (Română)</h2>
<p class="muted">Descompunere arhitecturală polifonică (Sopran, Alto, Tenor, Bas) și aliniere silabică conform tezei IDSR.</p>
<div style="font-size:12px;color:#34d399;background:rgba(52,211,153,0.1);border:1px solid #34d399;border-radius:8px;padding:8px;margin-bottom:12px">✅ Suport text & diacritice românești (ă, â, î, ș, ț) activat în UTF-8. Silabele și cratimele sunt legate direct de note.</div>
<div id="satb-details" style="display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px;margin-top:10px"></div>
</section>
<p class="muted">Only currently modeled staff lines, barlines and notehead proposals are drawn. This is not yet a pixel-identical replica; the source raster remains the identity anchor.</p>
<img id="geometry-render" alt="Structural render of currently modeled primitives">
<button id="render-current" type="button">Render current geometry</button>
</section>
<section class="panel"><h2>Lossless residual gate</h2>
<p class="muted">This proof covers exact recomposition of the decoded grayscale analysis raster, not symbolic correctness or physical ink depth. The original uploaded file remains separately preserved byte-for-byte.</p>
<pre id="identity-summary">Waiting for analysis</pre>
<button id="download-residual" type="button" disabled>Download uint16 residual PNG</button>
<button id="download-identity-report" type="button" disabled>Download identity report JSON</button>
</section>
<section class="panel" id="panel-4d-reader">
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:10px;margin-bottom:12px;">
  <div>
    <h2 style="margin:0;color:#58a6ff;display:flex;align-items:center;gap:8px">
      <span>🌐 Geamăn Digital 3D & Cititor Continuu 4D</span>
    </h2>
    <p class="muted" style="margin:4px 0 0 0">
      Reconstrucție fizică WebGL volumetrică (relief cerneală pe pergament). Cititorul optic 4D parcurge partitura în timp real și sintetizează frecvențele direct din geometria cernelii, ocolind complet OMR-ul.
    </p>
  </div>
  <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
    <button id="btn-play-4d" type="button" style="background:#238636;color:#ffffff;padding:10px 18px;border-radius:8px;font-size:14px;font-weight:700;display:inline-flex;align-items:center;gap:6px;cursor:pointer">
      ▶️ Pornește Cititorul 4D
    </button>
    <button id="btn-stop-4d" type="button" style="background:#21262d;color:#edf0f6;border:1px solid #303949;padding:10px 14px;border-radius:8px;font-size:14px;font-weight:600;cursor:pointer">
      ⏹️ Reset
    </button>
    <label style="display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#9da8b9">
      Tempo: <input id="reader4d-tempo" type="range" min="40" max="180" value="80" style="vertical-align:middle;width:90px"> <span id="reader4d-tempo-val">80 BPM</span>
    </label>
    <label for="lambda" style="display:inline-flex;align-items:center;gap:6px;font-size:13px;color:#9da8b9">
      Separare 3D λ: <input id="lambda" type="range" min="0" max="1" step="0.02" value="0.55" style="vertical-align:middle;width:80px">
    </label>
    <button id="reset-view" type="button" style="padding:6px 12px;font-size:12px;background:#303949;color:#edf0f6;border-radius:6px">Reset cameră</button>
  </div>
</div>
<div id="status-4d-bar" style="background:#0d1117;border:1px solid #303949;padding:10px 14px;border-radius:6px;margin-bottom:10px;font-family:monospace;font-size:13px;color:#38bdf8;display:flex;justify-content:space-between;align-items:center">
  <span id="status-4d-text">Cititor 4D pregătit. Apasă pe Play pentru scanare optică și sinteză directă din cerneală.</span>
  <span id="status-4d-metric">0 note active</span>
</div>
<div id="threejs-container" style="width:100%;height:460px;position:relative;border-radius:10px;overflow:hidden;background:#07090e;border:1px solid #303949;box-shadow:inset 0 0 40px rgba(0,0,0,0.8)">
  <canvas id="scene3d" width="1100" height="460" aria-label="Interactive semantic layer preview" style="width:100%;height:100%;display:block"></canvas>
</div>
<div style="display:flex;justify-content:space-between;align-items:center;margin-top:8px;font-size:12px;color:#6e7681">
  <span>Rotire: Click stânga + Drag | Pan: Click dreapta + Drag | Zoom: Scroll (poți privi notele inclusiv de dedesubt)</span>
  <span>Sinteză: Web Audio API Oscillator Matrix (Direct Y → Hz)</span>
</div>
</section>
<section class="panel"><h2>Review and edit pitch proposals</h2>
<p class="muted">Changing a pitch changes the symbolic proposal only. The original image and its recorded hash do not change. Durations, key signatures, accidentals, voices and part assignment still require review.</p>
<div id="editor"></div><button id="export-edits" type="button">Regenerate MusicXML from edits</button>
<button id="export-gltf-edits" type="button">Download current 3D scene (.gltf)</button>
</section>
<section class="panel"><h2>Warnings</h2><pre id="warnings"></pre></section>
</div>
<script>
let currentScene=null, originalScene=null, backgroundPreviewPngBase64=null, residualPngBase64=null, identityReportJson=null, yaw=-0.38, tilt=-0.2, dragging=false, draggingNote=-1, lastX=0, lastY=0;
let cornerImage=null, cornerScaleX=1, cornerScaleY=1, cornerPoints=[], selectingCorners=false, cornerObjectUrl=null;
const canvas=document.getElementById('scene3d'),ctx=canvas.getContext('2d');
const pageCanvas=document.getElementById('pageCanvas'),pageCtx=pageCanvas.getContext('2d');
const cornerCanvas=document.getElementById('cornerCanvas'),cornerCtx=cornerCanvas.getContext('2d');
const sourcePreview=new Image();
sourcePreview.onload=()=>{pageCanvas.width=sourcePreview.naturalWidth;pageCanvas.height=sourcePreview.naturalHeight;drawPage();};
function pagePoint(event){
 const rect=pageCanvas.getBoundingClientRect();
 return [(event.clientX-rect.left-pageCanvas.clientLeft)/pageCanvas.clientWidth*currentScene.image.width,
         (event.clientY-rect.top-pageCanvas.clientTop)/pageCanvas.clientHeight*currentScene.image.height];
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
 const showNotes=document.getElementById('show-notes')?.checked??true;
 const showLyrics=document.getElementById('show-lyrics')?.checked??true;
 const showSatb=document.getElementById('show-satb')?.checked??true;
 const showFingers=document.getElementById('show-fingers')?.checked??true;

 if(showSatb && currentScene.staffs){
  currentScene.staffs.forEach((st, sidx)=>{
   const vName = st.voice_name || (currentScene.systems?.[sidx]?.voice_name) || ('Portativ ' + (sidx+1));
   const clefInfo = st.clef || (currentScene.systems?.[sidx]?.inferred_clef) || {sign:'G', line:2};
   const clefLabel = clefInfo.sign + clefInfo.line;
   const badgeText = '🎼 ' + vName + ' [' + clefLabel + ']';
   const bx = 12 * sx, by = (st.top - 6) * sy;
   pageCtx.fillStyle = '#a78bfa';
   pageCtx.font = 'bold ' + Math.max(10, pageCanvas.width/90) + 'px system-ui';
   pageCtx.fillText(badgeText, bx, by > 14 ? by : (st.bottom + 14) * sy);
  });
 }

 if(showNotes){
  currentScene.notes.forEach((note,index)=>{
   const box=note.bbox,x=box[0]*sx,y=box[1]*sy,w=box[2]*sx,h=box[3]*sy;
   pageCtx.strokeStyle=index===draggingNote?'#b7f36b':'#e56f79';pageCtx.lineWidth=Math.max(1.4,pageCanvas.width/900);
   pageCtx.strokeRect(x,y,w,h);
   pageCtx.fillStyle='#e56f79';pageCtx.font=(Math.max(10,pageCanvas.width/100))+'px system-ui';
   pageCtx.fillText(note.pitch.step+note.pitch.octave+'?',x,y-3<10?y+h+12:y-3);
  });
 }

 currentScene.unknown_objects.forEach(item=>{
  const x=item.bbox[0]*sx,y=item.bbox[1]*sy,w=item.bbox[2]*sx,h=item.bbox[3]*sy;
  pageCtx.strokeStyle='#e3b35b';pageCtx.lineWidth=1;pageCtx.strokeRect(x,y,w,h);
 });

 if(showLyrics && currentScene.idsr_dynamic_layer && currentScene.idsr_dynamic_layer.dynamic_lyrics_detected){
  currentScene.idsr_dynamic_layer.dynamic_lyrics_detected.forEach(item=>{
   const lb = item.bbox_px, lx = lb[0]*sx, ly = lb[1]*sy, lw = lb[2]*sx, lh = lb[3]*sy;
   pageCtx.strokeStyle = '#34d399';
   pageCtx.lineWidth = 1.2;
   pageCtx.strokeRect(lx, ly, lw, lh);
   if(item.kind === 'lyric_hyphen'){
    pageCtx.fillStyle = '#34d399';
    pageCtx.font = 'bold ' + Math.max(10, pageCanvas.width/100) + 'px system-ui';
    pageCtx.fillText('-', lx + 2, ly + lh - 1);
   } else {
    pageCtx.fillStyle = '#34d399';
    pageCtx.font = (Math.max(8, pageCanvas.width/130)) + 'px system-ui';
    pageCtx.fillText('silabă', lx + 1, ly - 2 < 8 ? ly + lh + 9 : ly - 2);
   }
  });
 }

 if(showFingers && currentScene.finger_occlusions && currentScene.finger_occlusions.detected_fingers){
  currentScene.finger_occlusions.detected_fingers.forEach(f=>{
   const fb=f.bbox_px,fx=fb[0]*sx,fy=fb[1]*sy,fw=fb[2]*sx,fh=fb[3]*sy;
   pageCtx.strokeStyle='#38bdf8';pageCtx.lineWidth=2;pageCtx.setLineDash([4,3]);
   pageCtx.strokeRect(fx,fy,fw,fh);pageCtx.setLineDash([]);
   pageCtx.fillStyle='#38bdf8';pageCtx.font='bold '+(Math.max(9,pageCanvas.width/100))+'px system-ui';
   pageCtx.fillText('✋ '+f.finger_id+' (occlusion mask)',fx+4,fy+14);
  });
 }
}
function activeFile(){return document.getElementById('camera-file').files[0]||document.getElementById('file').files[0]||null;}
function drawCornerPage(){
 if(!cornerImage||!cornerImage.complete)return;
 cornerCtx.clearRect(0,0,cornerCanvas.width,cornerCanvas.height);
 cornerCtx.drawImage(cornerImage,0,0,cornerCanvas.width,cornerCanvas.height);
 if(cornerPoints.length>1){
  cornerCtx.strokeStyle='#d43853';cornerCtx.lineWidth=Math.max(2,cornerCanvas.width/600);cornerCtx.beginPath();
  cornerPoints.forEach((point,index)=>{const x=point[0]*cornerScaleX,y=point[1]*cornerScaleY;index===0?cornerCtx.moveTo(x,y):cornerCtx.lineTo(x,y);});
  if(cornerPoints.length===4)cornerCtx.closePath();cornerCtx.stroke();
 }
 cornerPoints.forEach((point,index)=>{
  const x=point[0]*cornerScaleX,y=point[1]*cornerScaleY;
  cornerCtx.fillStyle='#d43853';cornerCtx.beginPath();cornerCtx.arc(x,y,Math.max(8,cornerCanvas.width/150),0,Math.PI*2);cornerCtx.fill();
  cornerCtx.fillStyle='#fff';cornerCtx.font='bold '+Math.max(11,cornerCanvas.width/100)+'px system-ui';
  cornerCtx.fillText(String(index+1),x+5,y-5);
 });
}
function loadCornerPreview(){
 const file=activeFile();cornerPoints=[];selectingCorners=false;cornerImage=null;
 document.getElementById('corner-status').textContent='';
 if(cornerObjectUrl){URL.revokeObjectURL(cornerObjectUrl);cornerObjectUrl=null;}
 if(!file||file.name.toLowerCase().endsWith('.pdf')){document.getElementById('corner-panel').hidden=true;return;}
 cornerObjectUrl=URL.createObjectURL(file);cornerImage=new Image();
 cornerImage.onload=()=>{
  const previewScale=Math.min(1,1600/Math.max(cornerImage.naturalWidth,cornerImage.naturalHeight));
  cornerCanvas.width=Math.max(1,Math.round(cornerImage.naturalWidth*previewScale));
  cornerCanvas.height=Math.max(1,Math.round(cornerImage.naturalHeight*previewScale));
  cornerScaleX=cornerCanvas.width/cornerImage.naturalWidth;
  cornerScaleY=cornerCanvas.height/cornerImage.naturalHeight;
  document.getElementById('corner-panel').hidden=false;drawCornerPage();
  URL.revokeObjectURL(cornerObjectUrl);cornerObjectUrl=null;
 };
 cornerImage.onerror=()=>{document.getElementById('corner-panel').hidden=true;};
 cornerImage.src=cornerObjectUrl;
}
let threeRenderer=null, threeScene=null, threeCamera=null, threeControls=null;
let threeObjectsGroup=null, threeNoteMeshes={}, laserBeamMesh=null;
let audioCtx=null;

function getAudioContext(){
 if(!audioCtx){
  const AudioContextClass=window.AudioContext||window.webkitAudioContext;
  if(AudioContextClass)audioCtx=new AudioContextClass();
 }
 if(audioCtx&&audioCtx.state==='suspended')audioCtx.resume();
 return audioCtx;
}

function playOpticalTone(freq,durSec=0.38,gainLevel=0.22){
 const ctx=getAudioContext();if(!ctx)return;
 try{
  const now=ctx.currentTime;
  const osc=ctx.createOscillator();
  const oscHarm=ctx.createOscillator();
  const masterGain=ctx.createGain();

  osc.type='triangle';
  osc.frequency.setValueAtTime(freq,now);

  oscHarm.type='sine';
  oscHarm.frequency.setValueAtTime(freq*2.0,now);

  const harmGain=ctx.createGain();
  harmGain.gain.setValueAtTime(gainLevel*0.28,now);

  masterGain.gain.setValueAtTime(0.0001,now);
  masterGain.gain.linearRampToValueAtTime(gainLevel,now+0.015);
  masterGain.gain.exponentialRampToValueAtTime(gainLevel*0.65,now+0.08);
  masterGain.gain.exponentialRampToValueAtTime(0.0001,now+durSec);

  osc.connect(masterGain);
  oscHarm.connect(harmGain);
  harmGain.connect(masterGain);
  masterGain.connect(ctx.destination);

  osc.start(now);
  oscHarm.start(now);
  osc.stop(now+durSec);
  oscHarm.stop(now+durSec);
 }catch(e){console.warn('Audio synth warning:',e);}
}

const reader4D={
 running:false,tempoBpm:80,currentSystemIdx:0,playheadX:0,minX:0,maxX:1000,
 notesInActiveSystem:[],playedNoteIds:new Set(),animId:null,lastTimestamp:null,

 start(){
  if(!currentScene||!currentScene.notes||!currentScene.notes.length)return;
  getAudioContext();this.running=true;
  const btn=document.getElementById('btn-play-4d');
  if(btn){btn.textContent='⏸️ Pauză Cititor 4D';btn.style.background='#d97706';}
  if(this.playedNoteIds.size===0)this.initSystem(0);
  this.lastTimestamp=performance.now();
  this.loop=this.loop.bind(this);
  this.animId=requestAnimationFrame(this.loop);
 },

 pause(){
  this.running=false;
  const btn=document.getElementById('btn-play-4d');
  if(btn){btn.textContent='▶️ Reia Cititorul 4D';btn.style.background='#238636';}
  if(this.animId)cancelAnimationFrame(this.animId);
 },

 reset(){
  this.pause();this.currentSystemIdx=0;this.playedNoteIds.clear();
  const btn=document.getElementById('btn-play-4d');
  if(btn){btn.textContent='▶️ Pornește Cititorul 4D';btn.style.background='#238636';}
  if(currentScene)this.initSystem(0);
  const st=document.getElementById('status-4d-text');
  if(st)st.textContent='Cititor 4D resetat la începutul partiturii.';
 },

 initSystem(sysIdx){
  if(!currentScene||!currentScene.staffs)return;
  const numStaffs=currentScene.staffs.length;
  const isSatb2=(numStaffs%2===0&&numStaffs>=4);
  const activeStaves=isSatb2?[sysIdx*2,sysIdx*2+1]:[sysIdx];
  this.currentSystemIdx=sysIdx;
  this.notesInActiveSystem=currentScene.notes.filter(n=>activeStaves.includes(Number(n.staff_index)));
  if(this.notesInActiveSystem.length>0){
   const xs=this.notesInActiveSystem.map(n=>Number(n.center[0]));
   this.minX=Math.max(0,Math.min(...xs)-40);
   this.maxX=Math.max(...xs)+60;
  }else{
   this.minX=100;this.maxX=currentScene.image?currentScene.image.width-100:1000;
  }
  this.playheadX=this.minX;
  this.updateLaserPosition();
 },

 loop(timestamp){
  if(!this.running)return;
  const dt=Math.min(0.1,(timestamp-this.lastTimestamp)/1000);
  this.lastTimestamp=timestamp;
  const speed=(this.tempoBpm/60)*115;
  this.playheadX+=speed*dt;

  this.notesInActiveSystem.forEach(note=>{
   const nx=Number(note.center[0]);
   if(nx<=this.playheadX&&!this.playedNoteIds.has(note.id)){
    this.playedNoteIds.add(note.id);
    const freq=Number(note.frequency_hz)||(440*Math.pow(2,(Number(note.midi_pitch||69)-69)/12));
    playOpticalTone(freq,0.40);
    if(threeNoteMeshes[note.id]){
     const m=threeNoteMeshes[note.id];
     m.material.emissive.setHex(0x38bdf8);
     setTimeout(()=>{if(m&&m.material)m.material.emissive.setHex(0x000000);},280);
    }
    const st=document.getElementById('status-4d-text');
    if(st)st.textContent=`▶ Sistem ${this.currentSystemIdx+1} · Portativ ${Number(note.staff_index)+1} · Nota: ${note.pitch.step}${note.pitch.octave} (${freq.toFixed(1)} Hz)`;
   }
  });

  this.updateLaserPosition();

  if(this.playheadX>=this.maxX){
   const numStaffs=currentScene.staffs.length;
   const isSatb2=(numStaffs%2===0&&numStaffs>=4);
   const totalSystems=isSatb2?(numStaffs/2):numStaffs;
   if(this.currentSystemIdx+1<totalSystems){
    this.initSystem(this.currentSystemIdx+1);
   }else{
    this.reset();
    const st=document.getElementById('status-4d-text');
    if(st)st.textContent='✅ Scanare 4D completă pentru toată partitura!';
    return;
   }
  }
  this.animId=requestAnimationFrame(this.loop);
 },

 updateLaserPosition(){
  if(!currentScene)return;
  const w=currentScene.image.width,h=currentScene.image.height;
  if(laserBeamMesh){
   laserBeamMesh.position.x=this.playheadX-w/2;
   laserBeamMesh.visible=true;
   const numStaffs=currentScene.staffs.length;
   const isSatb2=(numStaffs%2===0&&numStaffs>=4);
   const activeStaves=isSatb2?[this.currentSystemIdx*2,this.currentSystemIdx*2+1]:[this.currentSystemIdx];
   const s0=currentScene.staffs[activeStaves[0]];
   const s1=currentScene.staffs[activeStaves[activeStaves.length-1]];
   if(s0&&s1){
    const topY=s0.top-20,botY=s1.bottom+20;
    const midY=(topY+botY)/2;
    laserBeamMesh.position.y=-(midY-h/2);
    laserBeamMesh.scale.y=(botY-topY)/200;
   }
  }
 }
};

function initThreeJs(){
 const container=document.getElementById('threejs-container');
 if(!container||!window.THREE)return false;
 if(threeRenderer)return true;
 try{
  const w=container.clientWidth||1100,h=container.clientHeight||460;
  threeScene=new THREE.Scene();
  threeScene.background=new THREE.Color(0x07090e);

  threeCamera=new THREE.PerspectiveCamera(45,w/h,1,15000);
  threeCamera.position.set(0,-400,1600);

  threeRenderer=new THREE.WebGLRenderer({antialias:true,alpha:true});
  threeRenderer.setSize(w,h);
  threeRenderer.setPixelRatio(Math.min(window.devicePixelRatio||1,2));
  threeRenderer.shadowMap.enabled=true;
  threeRenderer.shadowMap.type=THREE.PCFSoftShadowMap;

  const oldCanvas=document.getElementById('scene3d');
  if(oldCanvas&&oldCanvas.parentNode){
   oldCanvas.style.display='none';
   container.appendChild(threeRenderer.domElement);
  }

  if(window.THREE.OrbitControls){
   threeControls=new THREE.OrbitControls(threeCamera,threeRenderer.domElement);
   threeControls.enableDamping=true;
   threeControls.dampingFactor=0.08;
   threeControls.maxDistance=6000;
   threeControls.minDistance=200;
  }

  const amb=new THREE.AmbientLight(0xffffff,0.75);
  threeScene.add(amb);

  const dir=new THREE.DirectionalLight(0xfff6e8,0.85);
  dir.position.set(400,800,1200);
  dir.castShadow=true;
  threeScene.add(dir);

  const blueRim=new THREE.DirectionalLight(0x38bdf8,0.4);
  blueRim.position.set(-600,-800,400);
  threeScene.add(blueRim);

  threeObjectsGroup=new THREE.Group();
  threeScene.add(threeObjectsGroup);

  const laserGeo=new THREE.BoxGeometry(5,200,35);
  const laserMat=new THREE.MeshBasicMaterial({color:0x00f0ff,transparent:true,opacity:0.85});
  laserBeamMesh=new THREE.Mesh(laserGeo,laserMat);
  laserBeamMesh.visible=false;
  threeScene.add(laserBeamMesh);

  function animate(){
   requestAnimationFrame(animate);
   if(threeControls)threeControls.update();
   threeRenderer.render(threeScene,threeCamera);
  }
  animate();
  return true;
 }catch(e){
  console.error('Three.js init error:',e);
  return false;
 }
}

function buildThreeJsScene(){
 if(!initThreeJs()||!currentScene)return;
 while(threeObjectsGroup.children.length>0){
  threeObjectsGroup.remove(threeObjectsGroup.children[0]);
 }
 threeNoteMeshes={};
 const w=currentScene.image.width,h=currentScene.image.height;
 const sep=Number(document.getElementById('lambda').value)*Math.max(20,currentScene.staffs.length*12);

 // 1. Realistic 3D Paper Substrate
 const paperMat=new THREE.MeshStandardMaterial({color:0xfcf9f2,roughness:0.88,metalness:0.04});
 const paperMesh=new THREE.Mesh(new THREE.BoxGeometry(w,h,8),paperMat);
 paperMesh.receiveShadow=true;
 threeObjectsGroup.add(paperMesh);

 // 2. Staff Lines Extruded in 3D
 const staffLineMat=new THREE.MeshStandardMaterial({color:0x1e293b,roughness:0.4,metalness:0.2});
 currentScene.staffs.forEach(st=>{
  st.lines.forEach(ly=>{
   const lMesh=new THREE.Mesh(new THREE.BoxGeometry(w-40,2.2,2.0),staffLineMat);
   lMesh.position.set(0,-(ly-h/2),5.0);
   threeObjectsGroup.add(lMesh);
  });
  if(st.barlines){
   st.barlines.forEach(bx=>{
    const bMesh=new THREE.Mesh(new THREE.BoxGeometry(2.4,st.bottom-st.top,2.0),staffLineMat);
    bMesh.position.set(bx-w/2,-((st.top+st.bottom)/2-h/2),5.0);
    threeObjectsGroup.add(bMesh);
   });
  }
 });

 // 3. Volumetric 3D Noteheads with Carbon Ductus
 currentScene.notes.forEach(note=>{
  const nx=Number(note.center[0]),ny=Number(note.center[1]);
  const rx=Math.max(4.5,Number(note.bbox[2])*0.32);
  const ry=Math.max(3.2,Number(note.bbox[3])*0.26);
  const rz=Math.max(3.0,rx*0.6);
  const isHollow=note.shape_class==='hollow_head';
  const nMat=new THREE.MeshStandardMaterial({
   color:isHollow?0xf8fafc:0x0f172a,
   roughness:isHollow?0.8:0.35,
   metalness:isHollow?0.1:0.3,
   wireframe:isHollow
  });
  const nGeo=new THREE.SphereGeometry(1,16,12);
  const nMesh=new THREE.Mesh(nGeo,nMat);
  nMesh.scale.set(rx,ry,rz);
  nMesh.rotation.z=-0.35;
  nMesh.position.set(nx-w/2,-(ny-h/2),6.0+sep*0.25);
  nMesh.castShadow=true;
  threeObjectsGroup.add(nMesh);
  threeNoteMeshes[note.id]=nMesh;
 });

 // 4. Extruded 3D Stems and Beams
 const stemMat=new THREE.MeshStandardMaterial({color:0x111827,roughness:0.4,metalness:0.2});
 (currentScene.primitives||[]).filter(p=>p.kind==='stem').forEach(stem=>{
  const g=stem.geometry||{};
  const sx=Number(g.x0_px),sy0=Number(g.y0_px),sy1=Number(g.y1_px);
  const stemLen=Math.abs(sy1-sy0);
  const sMesh=new THREE.Mesh(new THREE.BoxGeometry(2.2,stemLen,2.5),stemMat);
  sMesh.position.set(sx-w/2,-((sy0+sy1)/2-h/2),6.0+sep*0.25);
  threeObjectsGroup.add(sMesh);
 });

 (currentScene.primitives||[]).filter(p=>p.kind==='beam').forEach(beam=>{
  const g=beam.geometry||{};
  const bx0=Number(g.x0_px),by0=Number(g.y0_px),bx1=Number(g.x1_px),by1=Number(g.y1_px);
  const bLen=Math.hypot(bx1-bx0,by1-by0);
  const bAngle=Math.atan2(-(by1-by0),bx1-bx0);
  const bMesh=new THREE.Mesh(new THREE.BoxGeometry(bLen,3.5,3.0),stemMat);
  bMesh.position.set((bx0+bx1)/2-w/2,-((by0+by1)/2-h/2),7.0+sep*0.35);
  bMesh.rotation.z=bAngle;
  threeObjectsGroup.add(bMesh);
 });

 document.getElementById('status-4d-metric').textContent=`${currentScene.notes.length} note active pe portative`;
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
 if(!currentScene)return;
 if(window.THREE)buildThreeJsScene();
 ctx.clearRect(0,0,canvas.width,canvas.height);
 ctx.fillStyle='#0b0f17';ctx.fillRect(0,0,canvas.width,canvas.height);
 const width=currentScene.image.width,height=currentScene.image.height;
 const sep=Number(document.getElementById('lambda').value)*Math.max(20,currentScene.staffs.length*12);
 const cx0=width/2,cy0=height/2;

 // 1. Render 3D Physical Paper Substrate in Perspective with Shadow
 const p00=projectPoint(0,0,0);
 const p10=projectPoint(width,0,0);
 const p11=projectPoint(width,height,0);
 const p01=projectPoint(0,height,0);

 // Soft paper drop shadow
 ctx.shadowColor='rgba(0,0,0,0.65)';
 ctx.shadowBlur=22;
 ctx.shadowOffsetX=6;
 ctx.shadowOffsetY=10;

 // Paper sheet surface (authentic ivory music score parchment)
 ctx.fillStyle='#faf8f5';
 ctx.beginPath();
 ctx.moveTo(p00[0],p00[1]);
 ctx.lineTo(p10[0],p10[1]);
 ctx.lineTo(p11[0],p11[1]);
 ctx.lineTo(p01[0],p01[1]);
 ctx.closePath();
 ctx.fill();

 // Reset shadow
 ctx.shadowColor='transparent';
 ctx.shadowBlur=0;
 ctx.shadowOffsetX=0;
 ctx.shadowOffsetY=0;

 // Paper trim border
 ctx.strokeStyle='#cbd5e1';
 ctx.lineWidth=1.5;
 ctx.stroke();

 // 2. Staff lines engraved upon the paper surface
 ctx.lineWidth=1.0;
 ctx.strokeStyle='#334155';
 currentScene.staffs.forEach(staff=>{
  staff.lines.forEach(y=>{
   const a=projectPoint(15,y,1);
   const b=projectPoint(width-15,y,1);
   ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  });
  if(staff.barlines){
   staff.barlines.forEach(bx=>{
    const a=projectPoint(bx,staff.top,1);
    const b=projectPoint(bx,staff.bottom,1);
    ctx.lineWidth=1.5;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
   });
  }
 });

 // 3. Vertical Stems in 3D
 const stemPrims=(currentScene.primitives||[]).filter(p=>p.kind==='stem');
 stemPrims.forEach(stem=>{
  const g=stem.geometry||{};
  const a=projectPoint(Number(g.x0_px),Number(g.y0_px),sep*0.35);
  const b=projectPoint(Number(g.x1_px),Number(g.y1_px),sep*0.35);
  ctx.strokeStyle='#1e293b';ctx.lineWidth=2.0;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
 });

 // 4. Realistic 3D Beams: Single Beams vs Dual Parallel Ribbons (Sixteenth notes: 2 bare de șaisprezecimi)
 const beamPrims=(currentScene.primitives||[]).filter(p=>p.kind==='beam');
 beamPrims.forEach(beam=>{
  const g=beam.geometry||{};
  const z=sep*0.55;
  const subBeams=g.sub_beams||[];
  ctx.fillStyle='#0f172a';
  ctx.strokeStyle='#0f172a';
  if(subBeams.length>1){
   // Explicit double/multi beams (sixteenth notes: 2 bare paralele)
   subBeams.forEach(sb=>{
    const sx0=sb[0],sy0=sb[1],sw=sb[2],sh=Math.max(2.5,sb[3]);
    const a=projectPoint(sx0,sy0,z);
    const b=projectPoint(sx0+sw,sy0,z);
    const c=projectPoint(sx0+sw,sy0+sh,z);
    const d=projectPoint(sx0,sy0+sh,z);
    ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.lineTo(c[0],c[1]);ctx.lineTo(d[0],d[1]);ctx.closePath();ctx.fill();
   });
  }else if(g.beam_count===2){
   // Synthesize two parallel ribbons
   const x0=Number(g.x0_px),y0=Number(g.y0_px),x1=Number(g.x1_px),y1=Number(g.y1_px);
   const thick=Math.max(2,Number(g.thickness_px_estimate||4)*0.45);
   const sepBeam=thick*1.6;
   const a1=projectPoint(x0,y0-sepBeam/2,z),b1=projectPoint(x1,y1-sepBeam/2,z);
   ctx.lineWidth=thick*1.2;ctx.beginPath();ctx.moveTo(a1[0],a1[1]);ctx.lineTo(b1[0],b1[1]);ctx.stroke();
   const a2=projectPoint(x0,y0+sepBeam/2,z),b2=projectPoint(x1,y1+sepBeam/2,z);
   ctx.lineWidth=thick*1.2;ctx.beginPath();ctx.moveTo(a2[0],a2[1]);ctx.lineTo(b2[0],b2[1]);ctx.stroke();
  }else{
   // Single beam ribbon (eighth note)
   const x0=Number(g.x0_px),y0=Number(g.y0_px),x1=Number(g.x1_px),y1=Number(g.y1_px);
   const thick=Math.max(3,Number(g.thickness_px_estimate||4));
   const a=projectPoint(x0,y0,z),b=projectPoint(x1,y1,z);
   ctx.lineWidth=thick*1.2;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
  }
 });

 // 5. Noteheads in 3D: Rotated Ellipses with Proper Ductus Angle
 currentScene.notes.forEach(note=>{
  const x=Number(note.center[0]),y=Number(note.center[1]);
  const z=sep*0.35;
  const p=projectPoint(x,y,z);
  const rx=Math.max(4,Number(note.bbox[2])*0.32);
  const ry=Math.max(3,Number(note.bbox[3])*0.26);
  const rotRad=(Number(note.geometry_refinement?.rotation_deg||-20))*Math.PI/180;
  ctx.save();
  ctx.translate(p[0],p[1]);
  ctx.rotate(rotRad+yaw*0.1);
  ctx.fillStyle='#0f172a';
  ctx.beginPath();
  ctx.ellipse(0,0,rx*1.2,ry,0,0,Math.PI*2);
  if(note.shape_class==='hollow_head'){
   ctx.lineWidth=2.2;ctx.strokeStyle='#0f172a';ctx.stroke();
  }else{
   ctx.fill();
  }
  ctx.restore();
  ctx.fillStyle='#475569';ctx.font='10px system-ui';
  ctx.fillText(note.pitch.step+note.pitch.octave,p[0]+rx+2,p[1]-ry);
 });

 // 6. True Augmentation Dots
 (currentScene.dots||[]).forEach(dot=>{
  const p=projectPoint(Number(dot.center[0]),Number(dot.center[1]),sep*0.35);
  ctx.fillStyle='#0f172a';ctx.beginPath();ctx.arc(p[0],p[1],2.5,0,Math.PI*2);ctx.fill();
 });

 // 7. Lyrics Layer (Real syllable badges, not toy squares!)
 const lyricsPrims=(currentScene.primitives||[]).filter(p=>p.kind==='lyric_syllable'||p.kind==='lyric_hyphen');
 lyricsPrims.forEach(lyr=>{
  const g=lyr.geometry||{};
  const bx=Number(g.center_x_px||(g.bbox_px&&g.bbox_px[0])||0);
  const by=Number(g.center_y_px||(g.bbox_px&&g.bbox_px[1])||0);
  const p=projectPoint(bx,by,sep*0.75);
  const text=lyr.kind==='lyric_hyphen'?'-':(lyr.text||'text');
  ctx.fillStyle='rgba(16,185,129,0.18)';
  ctx.strokeStyle='#10b981';
  ctx.lineWidth=1;
  const textW=ctx.measureText(text).width||16;
  ctx.beginPath();
  ctx.roundRect?ctx.roundRect(p[0]-textW/2-4,p[1]-8,textW+8,16,4):ctx.rect(p[0]-textW/2-4,p[1]-8,textW+8,16);
  ctx.fill();ctx.stroke();
  ctx.fillStyle='#059669';ctx.font='bold 10px system-ui';
  ctx.fillText(text,p[0]-textW/2,p[1]+4);
 });

 // 8. Dynamics Layer (Musical badges)
 const dynPrims=(currentScene.primitives||[]).filter(p=>(p.kind&&p.kind.startsWith('dynamic'))||(p.kind&&p.kind.startsWith('hairpin')));
 dynPrims.forEach(dyn=>{
  const g=dyn.geometry||{};
  const bx=Number((g.bbox_px&&g.bbox_px[0])||0)+10;
  const by=Number((g.bbox_px&&g.bbox_px[1])||0);
  const p=projectPoint(bx,by,sep*0.95);
  const glyph=dyn.kind.includes('mf')?'mf':(dyn.kind.includes('f')?'f':'p');
  ctx.fillStyle='#f59e0b';ctx.font='italic bold 12px serif';
  ctx.fillText(glyph,p[0]-6,p[1]+4);
 });

 // 9. Volta Brackets
 (currentScene.primitives||[]).filter(p=>p.kind==='volta_bracket').forEach(vlt=>{
  const g=vlt.geometry||{};const vx0=Number(g.vx0||(g.bbox_px&&g.bbox_px[0])||0);const vx1=Number(g.vx1||(vx0+60));const ry=Number(g.roof_y||(g.bbox_px&&g.bbox_px[1])||0);
  const a=projectPoint(vx0,ry,sep*1.30);const b=projectPoint(vx1,ry,sep*1.30);
  ctx.strokeStyle='#c678dd';ctx.lineWidth=2.0;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
 });

 ctx.fillStyle='#94a3b8';ctx.font='12px system-ui';
 ctx.fillText('ScoreTwin 3D: Substrat Hârtie · Portative · Note · Bare Paralele Șaisprezecimi · Versuri',16,24);
}
function buildEditor(){
 const holder=document.getElementById('editor');holder.replaceChildren();
 if(!currentScene||!currentScene.notes.length){holder.textContent='No notehead proposals are ready for pitch editing.';return;}
 const letters=['C','D','E','F','G','A','B'];
 currentScene.notes.forEach((note,index)=>{
  const row=document.createElement('div');row.className='editor-row';
  const hint=(note.duration_hypothesis&&note.duration_hypothesis.candidate)||'duration unresolved';
  const title=document.createElement('span');title.textContent='Proposal '+(index+1)+' · '+note.id+' · '+Math.round(note.center[0])+','+Math.round(note.center[1])+' · '+hint;
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
document.getElementById('file').addEventListener('change',loadCornerPreview);
document.getElementById('camera-file').addEventListener('change',loadCornerPreview);
document.getElementById('mark-corners').addEventListener('click',()=>{
 cornerPoints=[];selectingCorners=true;drawCornerPage();
 document.getElementById('corner-status').textContent='Tap four corners clockwise, starting at the top left.';
});
document.getElementById('clear-corners').addEventListener('click',()=>{
 cornerPoints=[];selectingCorners=false;drawCornerPage();
 document.getElementById('corner-status').textContent='Perspective correction cleared.';
});
cornerCanvas.addEventListener('pointerdown',event=>{
 if(!selectingCorners||!cornerImage)return;
 const rect=cornerCanvas.getBoundingClientRect();
 const x=(event.clientX-rect.left-cornerCanvas.clientLeft)/cornerCanvas.clientWidth*cornerCanvas.width/cornerScaleX;
 const y=(event.clientY-rect.top-cornerCanvas.clientTop)/cornerCanvas.clientHeight*cornerCanvas.height/cornerScaleY;
 cornerPoints.push([Math.max(0,Math.min(cornerImage.naturalWidth-1,x)),Math.max(0,Math.min(cornerImage.naturalHeight-1,y))]);
 drawCornerPage();
 if(cornerPoints.length===4){selectingCorners=false;document.getElementById('corner-status').textContent='Four corners selected. Analyze page will rectify this planar region.';}
 else document.getElementById('corner-status').textContent='Corner '+cornerPoints.length+' of 4 selected.';
});
form.addEventListener('submit',async(event)=>{
 event.preventDefault();const file=document.getElementById('camera-file').files[0]||document.getElementById('file').files[0];if(!file){document.getElementById('status').textContent='Choose a file or take a photo first.';return;}
 const button=document.getElementById('submit');button.disabled=true;const status=document.getElementById('status');status.textContent='Processing locally…';
 try{
  const body=new FormData();body.append('file',file);body.append('page',document.getElementById('page').value||'0');
  if(selectingCorners){status.textContent='Select all four page corners or clear the selection first.';button.disabled=false;return;}
  if(cornerPoints.length===4)body.append('corners_json',JSON.stringify(cornerPoints));
  const autoSub=document.getElementById('auto-substrate');
  if(autoSub)body.append('auto_substrate',autoSub.checked?'true':'false');
  const response=await fetch('/api/analyze',{method:'POST',body});const result=await response.json();
  if(!response.ok)throw new Error(result.detail||'Analysis failed');
  currentScene=result.scene;
  currentScene.source=result.source;
  originalScene=JSON.parse(JSON.stringify(currentScene));
  backgroundPreviewPngBase64=result.source_preview_png_base64;
  residualPngBase64=result.residual_png_base64||null;
  identityReportJson=result.identity_report_json||JSON.stringify(result.identity_reconstruction||{},null,2);
  sourcePreview.src='data:image/png;base64,'+result.source_preview_png_base64;
  document.getElementById('summary').textContent=JSON.stringify(result.summary,null,2);
  const unresolved=(currentScene.unknown_objects||[]).map(item=>item.id+': '+item.reason);
  document.getElementById('warnings').textContent=[...result.warnings,...unresolved].join('\n');
  document.getElementById('geometry-render').src='data:image/png;base64,'+result.geometry_render_png_base64;
  const proof=result.identity_reconstruction||{};
  document.getElementById('identity-summary').textContent=[
   'Exact pixel recomposition: '+(proof.exact_pixel_reconstruction_verified?'PASS':'FAIL'),
   'Mismatched pixels: '+(proof.mismatched_pixel_count??'not reported'),
   'Maximum absolute pixel error: '+(proof.maximum_absolute_pixel_error??'not reported'),
   'Raster pixels: '+(proof.pixel_count??'not reported'),
   'Residual PNG: '+(result.residual_download_available?'available for direct download':'kept in the full local bundle; too large for browser inline delivery'),
   'Scope: '+(proof.applies_to||'decoded analysis raster only')
  ].join('\\n');
  document.getElementById('download-residual').disabled=!residualPngBase64;
  document.getElementById('download-identity-report').disabled=!identityReportJson;
  setDownload(result.musicxml);
  setPdfDownload(result.digital_pdf_base64);
  setAudioDownloads(result.audio_mp3_base64, result.midi_base64);
  document.getElementById('results').hidden=false;
  buildEditor();draw3d();renderSatbPanel(currentScene);
  status.textContent='Analysis finished. Review every pitch and all warnings before using the MusicXML.';
 }catch(error){status.textContent=error.message||String(error);}
 finally{button.disabled=false;}
});
let digitalPdfBase64 = null;
function setPdfDownload(base64Data){
 digitalPdfBase64 = base64Data;
 const anchor = document.getElementById('download-digital-pdf');
 if(!anchor) return;
 if(base64Data){
  const binaryString = atob(base64Data);
  const bytes = new Uint8Array(binaryString.length);
  for (let i = 0; i < binaryString.length; i++) {
   bytes[i] = binaryString.charCodeAt(i);
  }
  const blob = new Blob([bytes], {type: 'application/pdf'});
  const url = URL.createObjectURL(blob);
  if(anchor.dataset.url) URL.revokeObjectURL(anchor.dataset.url);
  anchor.href = url;
  anchor.dataset.url = url;
 }
}
document.getElementById('download-digital-pdf').addEventListener('click', async (event) => {
 if(!digitalPdfBase64 && currentScene){
  event.preventDefault();
  const btn = document.getElementById('download-digital-pdf');
  const oldText = btn.textContent;
  btn.textContent = '⏳ Compiling vectorized PDF...';
  try {
   const response = await fetch('/api/export-pdf', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(currentScene)
   });
   if(!response.ok) throw new Error('PDF export failed');
   const blob = await response.blob();
   const url = URL.createObjectURL(blob);
   const a = document.createElement('a');
   a.href = url;
   a.download = 'score_digital.pdf';
   a.click();
   setTimeout(() => URL.revokeObjectURL(url), 1000);
  } catch(err) {
   alert('PDF generation error: ' + err.message);
  } finally {
   btn.textContent = oldText;
  }
 }
});
function setDownload(xml){
 const url=URL.createObjectURL(new Blob([xml],{type:'application/vnd.recordare.musicxml+xml'}));
 const anchor=document.getElementById('download');if(anchor.dataset.url)URL.revokeObjectURL(anchor.dataset.url);
 anchor.href=url;anchor.dataset.url=url;
}
function setAudioDownloads(mp3Base64, midiBase64){
 const player = document.getElementById('score-audio-player');
 const mp3Anchor = document.getElementById('download-audio-mp3');
 const midiAnchor = document.getElementById('download-midi');
 if(mp3Base64){
  const mp3Url = 'data:audio/mp3;base64,' + mp3Base64;
  player.src = mp3Url;
  player.hidden = false;
  if(mp3Anchor){
   mp3Anchor.href = mp3Url;
   mp3Anchor.hidden = false;
  }
 }
 if(midiBase64 && midiAnchor){
  const midiUrl = 'data:audio/midi;base64,' + midiBase64;
  midiAnchor.href = midiUrl;
  midiAnchor.hidden = false;
 }
}
function renderSatbPanel(scene){
 const panel = document.getElementById('satb-panel');
 const container = document.getElementById('satb-details');
 if(!panel || !container || !scene || !scene.staffs) return;
 container.innerHTML = '';
 const lyrics = (scene.idsr_dynamic_layer && scene.idsr_dynamic_layer.dynamic_lyrics_detected) || [];
 scene.staffs.forEach((st, idx)=>{
  const vName = st.voice_name || scene.systems?.[idx]?.voice_name || ('Portativ ' + (idx + 1));
  const clef = st.clef || scene.systems?.[idx]?.inferred_clef || {sign:'G', line:2};
  const staffNotes = (scene.notes || []).filter(n=>n.staff_index === idx);
  const staffLyrics = lyrics.filter(l=>l.staff_index === idx);
  const card = document.createElement('div');
  card.style = 'background:#10131a;border:1px solid #303949;border-radius:10px;padding:12px';
  card.innerHTML = `
    <div style="font-weight:700;color:#a78bfa;font-size:15px;margin-bottom:4px">🎵 ${vName}</div>
    <div style="font-size:12px;color:#9da8b9;margin-bottom:8px">Cheie: <b style="color:#edf0f6">${clef.sign}${clef.line}</b> (${clef.clef_id || 'standard'})</div>
    <div style="font-size:12px;color:#ced7e5">Note recunoscute: <b>${staffNotes.length}</b></div>
    <div style="font-size:12px;color:#34d399">Silabe / cratime text: <b>${staffLyrics.length}</b></div>
  `;
  container.appendChild(card);
 });
 panel.hidden = false;
}
['show-notes','show-lyrics','show-satb','show-fingers'].forEach(id=>{
 const el=document.getElementById(id);
 if(el) el.addEventListener('change', drawPage);
});
document.getElementById('lambda').addEventListener('input',()=>{draw3d();});
document.getElementById('reset-view').addEventListener('click',()=>{
 yaw=-0.38;tilt=-0.2;
 if(threeControls&&threeCamera){threeCamera.position.set(0,-400,1600);threeControls.reset();}
 draw3d();
});
const btnPlay4D = document.getElementById('btn-play-4d');
if(btnPlay4D) btnPlay4D.addEventListener('click', ()=>{reader4D.running ? reader4D.pause() : reader4D.start();});
const btnStop4D = document.getElementById('btn-stop-4d');
if(btnStop4D) btnStop4D.addEventListener('click', ()=>{reader4D.reset();});
const tempoSlider = document.getElementById('reader4d-tempo');
if(tempoSlider) tempoSlider.addEventListener('input', (e)=>{
 reader4D.tempoBpm = Number(e.target.value);
 const valSpan = document.getElementById('reader4d-tempo-val');
 if(valSpan) valSpan.textContent = e.target.value + ' BPM';
});
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
document.getElementById('download-residual').addEventListener('click',()=>{
 if(!residualPngBase64)return;
 const binary=atob(residualPngBase64),bytes=new Uint8Array(binary.length);
 for(let i=0;i<binary.length;i++)bytes[i]=binary.charCodeAt(i);
 const blob=new Blob([bytes],{type:'image/png'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
 anchor.href=url;anchor.download='residual-offset255.png';anchor.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
});
document.getElementById('download-identity-report').addEventListener('click',()=>{
 if(!identityReportJson)return;
 const blob=new Blob([identityReportJson],{type:'application/json'}),url=URL.createObjectURL(blob),anchor=document.createElement('a');
 anchor.href=url;anchor.download='identity-reconstruction.json';anchor.click();
 setTimeout(()=>URL.revokeObjectURL(url),1000);
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
document.getElementById('export-gltf-edits').addEventListener('click',async()=>{
 if(!currentScene)return;
 const button=document.getElementById('export-gltf-edits');button.disabled=true;
 try{
  const response=await fetch('/api/export-gltf',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({scene:currentScene,background_png_base64:backgroundPreviewPngBase64})});
  const result=await response.json();if(!response.ok)throw new Error(result.detail||'glTF export failed');
  const blob=new Blob([result.gltf],{type:'model/gltf+json'});
  const url=URL.createObjectURL(blob),anchor=document.createElement('a');
  anchor.href=url;anchor.download='scoretwin-scene.gltf';anchor.click();
  setTimeout(()=>URL.revokeObjectURL(url),1000);
  document.getElementById('status').textContent='Self-contained glTF exported with the analysis-raster texture. Z offsets are semantic layers, not physical ink depth.';
 }catch(error){document.getElementById('status').textContent=error.message||String(error);}
 finally{button.disabled=false;}
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
async def analyze(
    file: UploadFile = File(...),
    page: int = Form(0),
    corners_json: str | None = Form(None),
    auto_substrate: bool = Form(True),
) -> dict:
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
        corners = json.loads(corners_json) if corners_json else None
        if corners is not None and not isinstance(corners, list):
            raise ValueError("corners_json must be a JSON array of four [x, y] points.")
        with tempfile.TemporaryDirectory(prefix="scoretwin-") as directory:
            input_path = Path(directory) / ("source" + suffix)
            input_path.write_bytes(contents)
            page_index = max(0, page - 1)
            gray = load_grayscale(input_path, page=page_index)
            bundle_dir = Path(directory) / "bundle"
            model_path = Path(MODEL_PATH) if MODEL_PATH and Path(MODEL_PATH).exists() else None
            analysis_result = process_file(
                input_path,
                bundle_dir,
                page=page_index,
                model_path=model_path,
                rectify_corners=corners,
                auto_substrate=auto_substrate,
            )
            scene = analysis_result["scene"]
            source_sha256 = analysis_result["source_sha256"]
            scene["source"] = {
                **scene["source"],
                "filename": filename,
                "server_persisted": False,
            }
            analysis_pixels_path = analysis_result["analysis_pixels_path"]
            analysis_pixels = cv2.imread(str(analysis_pixels_path), cv2.IMREAD_UNCHANGED)
            if analysis_pixels is None:
                raise RuntimeError("Could not read the channel-preserving analysis raster.")
            preview = analysis_pixels
            preview_scale = min(1.0, 1600.0 / max(analysis_pixels.shape[1], analysis_pixels.shape[0]))
            if preview_scale < 1.0:
                preview = cv2.resize(
                    analysis_pixels,
                    (max(1, int(round(analysis_pixels.shape[1] * preview_scale))),
                     max(1, int(round(analysis_pixels.shape[0] * preview_scale)))),
                    interpolation=cv2.INTER_AREA,
                )
            success, encoded = cv2.imencode(".png", preview)
            if not success:
                raise RuntimeError("Preview encoding failed.")
            xml = Path(analysis_result["musicxml_path"]).read_text(encoding="utf-8")
            identity_report_json = Path(analysis_result["identity_report_path"]).read_text(encoding="utf-8")
            residual_bytes = Path(analysis_result["residual_path"]).read_bytes()
            residual_download_available = len(residual_bytes) <= 12 * 1024 * 1024
            residual_png_base64 = (
                base64.b64encode(residual_bytes).decode("ascii")
                if residual_download_available else None
            )
            digital_pdf_path = analysis_result.get("digital_pdf_path")
            digital_pdf_base64 = None
            if digital_pdf_path and Path(digital_pdf_path).exists():
                digital_pdf_base64 = base64.b64encode(Path(digital_pdf_path).read_bytes()).decode("ascii")
            midi_path = analysis_result.get("midi_path")
            midi_base64 = None
            if midi_path and Path(midi_path).exists():
                midi_base64 = base64.b64encode(Path(midi_path).read_bytes()).decode("ascii")
            mp3_path = analysis_result.get("mp3_path")
            audio_mp3_base64 = None
            if mp3_path and Path(mp3_path).exists():
                audio_mp3_base64 = base64.b64encode(Path(mp3_path).read_bytes()).decode("ascii")
            return {
                "summary": {
                    "filename": filename,
                    "page_index": page if suffix == ".pdf" else None,
                    "source_sha256": source_sha256,
                    "width": scene["image"]["width"],
                    "height": scene["image"]["height"],
                    "staff_count": len(scene["staffs"]),
                    "proposals": scene["proposal_count"],
                    "notehead_proposals": scene["note_count"],
                    "finger_occlusions_count": (scene.get("finger_occlusions") or {}).get("fingers_detected_count", 0),
                    "dynamic_lyrics_count": len((scene.get("idsr_dynamic_layer") or {}).get("dynamic_lyrics_detected", [])),
                    "dynamic_dynamics_count": len((scene.get("idsr_dynamic_layer") or {}).get("dynamic_dynamics_detected", [])),
                    "dynamic_voltas_count": len((scene.get("idsr_dynamic_layer") or {}).get("dynamic_volta_brackets_detected", [])),
                    "investigated_traces_count": len((scene.get("idsr_dynamic_layer") or {}).get("investigated_traces", [])),
                    "zero_abstentions_verified": bool((scene.get("idsr_dynamic_layer") or {}).get("abstained_or_unknown_count", 0) == 0),
                    "runtime_seconds": scene["runtime_seconds"],
                },
                "warnings": scene["warnings"],
                "source": scene["source"],
                "scene": scene,
                "musicxml": xml,
                "digital_pdf_base64": digital_pdf_base64,
                "audio_mp3_base64": audio_mp3_base64,
                "midi_base64": midi_base64,
                "identity_reconstruction": scene.get("identity_reconstruction", {}),
                "identity_report_json": identity_report_json,
                "residual_png_base64": residual_png_base64,
                "residual_download_available": residual_download_available,
                "residual_size_bytes": len(residual_bytes),
                "source_preview_png_base64": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "geometry_render_png_base64": _render_base64(
                    (
                        max(1, int(round(analysis_pixels.shape[0] * preview_scale))),
                        max(1, int(round(analysis_pixels.shape[1] * preview_scale))),
                    ),
                    scene,
                    scale=preview_scale,
                ),
            }
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/export-pdf")
def export_pdf(scene: dict = Body(...)) -> Response:
    """Compile digitalized vectorized PDF directly from the current scene."""
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise HTTPException(status_code=400, detail="Unrecognized ScoreTwin scene schema.")
    with tempfile.TemporaryDirectory() as td:
        pdf_out = Path(td) / "score_digital.pdf"
        res = compile_digital_score_pdf(scene, pdf_out)
        if not res or not pdf_out.exists():
            raise HTTPException(status_code=500, detail="Could not compile digital PDF.")
        return Response(
            content=pdf_out.read_bytes(),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment; filename=score_digital.pdf"},
        )


@app.post("/api/export-audio")
def export_audio(scene: dict = Body(...)) -> dict:
    """Compile digitalized audio (MP3 and MIDI) directly from the current scene."""
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise HTTPException(status_code=400, detail="Unrecognized ScoreTwin scene schema.")
    with tempfile.TemporaryDirectory() as td:
        midi_out = Path(td) / "score_audio.mid"
        mp3_out = Path(td) / "score_audio.mp3"
        midi_res, mp3_res = compile_digital_score_audio(scene, midi_out, mp3_out)
        return {
            "midi_base64": base64.b64encode(midi_out.read_bytes()).decode("ascii") if midi_res and midi_out.exists() else None,
            "audio_mp3_base64": base64.b64encode(mp3_out.read_bytes()).decode("ascii") if mp3_res and mp3_out.exists() else None,
        }


def _render_base64(shape: tuple[int, int], scene: dict, scale: float = 1.0) -> str:
    rendered = render_geometry_scene(shape, scene, scale=scale)
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
    scale = min(1.0, 1600.0 / max(height, width))
    output_shape = (max(1, int(round(height * scale))), max(1, int(round(width * scale))))
    try:
        return {"geometry_render_png_base64": _render_base64(output_shape, scene, scale=scale)}
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.post("/api/export-gltf")
def export_gltf(payload: dict = Body(...)) -> dict:
    scene = payload.get("scene", payload)
    background_png = None
    encoded_background = payload.get("background_png_base64") if "scene" in payload else None
    if encoded_background is not None:
        if not isinstance(encoded_background, str) or len(encoded_background) > 32 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Background preview payload is invalid or too large.")
        try:
            background_png = base64.b64decode(encoded_background, validate=True)
        except (ValueError, TypeError) as exc:
            raise HTTPException(status_code=422, detail="Background preview must be base64-encoded PNG data.") from exc
        if not background_png.startswith(b"\x89PNG\r\n\x1a\n"):
            raise HTTPException(status_code=422, detail="Background preview must be a PNG.")
    if scene.get("schema") != "scoretwin-scene-0.1":
        raise HTTPException(status_code=400, detail="Unrecognized ScoreTwin scene schema.")
    if len(scene.get("notes", [])) > 5000 or len(scene.get("primitives", [])) > 30000:
        raise HTTPException(status_code=413, detail="Scene contains too many notes or geometry primitives.")
    try:
        gltf = build_gltf(scene, background_png=background_png)
        validation = validate_embedded_gltf(gltf)
    except (ValueError, TypeError, KeyError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if len(gltf.encode("utf-8")) > 45 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="The self-contained glTF export exceeds 45 MiB.")
    return {"gltf": gltf, "validation": validation}


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
