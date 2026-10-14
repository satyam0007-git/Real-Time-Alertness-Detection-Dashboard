import os
import uuid
import json
import time
import shutil
from typing import Optional
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2

from detector import VideoAlertnessProcessor, EyeStateDetector, AlertnessTracker

app = FastAPI(
    title="Real-Time Alertness Detection API",
    description="FastAPI service for drowsiness & sleep alertness detection",
    version="1.0.0"
)

# Enable CORS for any frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files with native HTTP Range support (prevents browser video player from getting stuck)
app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

# Copy sample video into uploads if available
default_sample = os.path.join(BASE_DIR, "satyam.mp4")
if os.path.exists(default_sample):
    dest_sample = os.path.join(UPLOAD_DIR, "sample-demo.mp4")
    if not os.path.exists(dest_sample):
        shutil.copyfile(default_sample, dest_sample)

processor = VideoAlertnessProcessor()
live_detector = EyeStateDetector()


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accepts video file, saves to disk, returns unique video_id."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    extension = os.path.splitext(file.filename)[1] or ".mp4"
    video_id = f"vid-{uuid.uuid4().hex[:8]}"
    save_path = os.path.join(UPLOAD_DIR, f"{video_id}{extension}")

    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):
                f.write(chunk)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {str(e)}")

    return {
        "video_id": video_id,
        "filename": file.filename,
        "video_url": f"/uploads/{video_id}{extension}"
    }


@app.get("/detect/{video_id}")
async def detect_alertness(video_id: str):
    """
    Runs alertness detection on video (or returns cached results).
    Returns scores, statuses, timeline, bounding boxes, and summary stats.
    """
    video_filename = None
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        cand = f"{video_id}{ext}"
        if os.path.exists(os.path.join(UPLOAD_DIR, cand)):
            video_filename = cand
            break

    if not video_filename and os.path.exists(os.path.join(UPLOAD_DIR, video_id)):
        video_filename = video_id

    if not video_filename:
        raise HTTPException(status_code=404, detail=f"Video '{video_id}' not found")

    video_path = os.path.join(UPLOAD_DIR, video_filename)

    # Check cache for instant sub-second response
    cache_path = os.path.join(UPLOAD_DIR, f"{video_id}_results.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r") as f:
            data = json.load(f)
            data["video_url"] = f"/uploads/{video_filename}"
            return data

    try:
        analysis = processor.process_video_file(video_path, sample_fps=4.0)
        scores = [item["score"] for item in analysis["results"]]
        statuses = [item["status"] for item in analysis["results"]]

        response_data = {
            "video_id": video_id,
            "video_url": f"/uploads/{video_filename}",
            "scores": scores,
            "statuses": statuses,
            "summary": analysis["summary"],
            "results": analysis["results"]
        }

        with open(cache_path, "w") as f:
            json.dump(response_data, f, indent=2)

        return response_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection error: {str(e)}")


@app.get("/stream/{video_id}")
@app.get("/video/{video_id}")
async def stream_video_redirect(video_id: str):
    """Redirect to range-supporting static file mount."""
    for ext in [".mp4", ".avi", ".mov", ".mkv", ".webm"]:
        cand = f"{video_id}{ext}"
        if os.path.exists(os.path.join(UPLOAD_DIR, cand)):
            from fastapi.responses import RedirectResponse
            return RedirectResponse(url=f"/uploads/{cand}")

    if os.path.exists(os.path.join(UPLOAD_DIR, video_id)):
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url=f"/uploads/{video_id}")

    raise HTTPException(status_code=404, detail="Video not found")


# Shared state for live webcam tracking
latest_webcam_state = {
    "active": False,
    "timestamp": 0.0,
    "status": "awake",
    "score": 100.0,
    "closed_duration": 0.0,
    "ear": 0.35,
    "face_detected": False,
    "frame_count": 0
}


def generate_webcam_stream():
    """Generator for live webcam feed with safe fallback and state tracking."""
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    # Request widescreen 720p resolution for crisp, full-frame capture
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        blank = np.zeros((360, 640, 3), dtype=np.uint8)
        cv2.putText(blank, "Webcam Not Accessible", (120, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', blank)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        return

    tracker = AlertnessTracker(drowsy_threshold_sec=1.0, sleep_threshold_sec=3.0)
    start_time = time.time()
    frame_idx = 0
    latest_webcam_state["active"] = True

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            current_time = round(time.time() - start_time, 2)
            det = live_detector.detect(frame)
            status, score, closed_dur = tracker.update(det["is_eyes_closed"], current_time)

            latest_webcam_state.update({
                "active": True,
                "timestamp": current_time,
                "status": status,
                "score": score,
                "closed_duration": closed_dur,
                "ear": det["ear_score"],
                "face_detected": det["face_detected"],
                "frame_count": frame_idx
            })

            if det["face_detected"] and det["bbox"]:
                x, y, w, h = det["bbox"]
                color = (0, 255, 0) if status == "awake" else ((0, 255, 255) if status == "drowsy" else (0, 0, 255))
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                badge_text = f"{status.upper()} ({int(score)}%)"
                sub_text = f"Closed: {closed_dur:.1f}s | EAR: {det['ear_score']}"
                cv2.rectangle(frame, (x, max(0, y - 28)), (x + 200, y), color, -1)
                cv2.putText(frame, badge_text, (x + 5, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                cv2.putText(frame, sub_text, (x, min(frame.shape[0] - 10, y + h + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
    finally:
        latest_webcam_state["active"] = False
        cap.release()


@app.get("/webcam_status")
async def get_webcam_status():
    """Returns the latest real-time telemetry from the live webcam."""
    return latest_webcam_state


@app.get("/video_feed")
def video_feed():
    """Live MJPEG webcam stream with real-time detection overlay."""
    return StreamingResponse(
        generate_webcam_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/", response_class=HTMLResponse)
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Ultra-responsive native dashboard (zero in-browser Babel lag)."""
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Real-Time Alertness Detection Dashboard</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body { background-color: #020617; color: #f8fafc; font-family: system-ui, -apple-system, sans-serif; }
    .badge-awake { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }
    .badge-drowsy { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }
    .badge-sleeping { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
    /* Custom thin scrollbar for right-hand table */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #0b1120; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #475569; }
  </style>
</head>
<body class="h-screen w-screen overflow-hidden flex flex-col p-2.5 md:p-3">

  <!-- 1. Compact Header Bar -->
  <header class="flex items-center justify-between pb-2 border-b border-slate-800 shrink-0">
    <div class="flex items-center gap-2.5">
      <div class="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center">
        <i class="fa-solid fa-eye text-emerald-400 text-sm"></i>
      </div>
      <div>
        <h1 class="text-sm font-bold text-white tracking-tight leading-tight">Real-Time Alertness Detection</h1>
        <p class="text-[10px] text-slate-400 leading-tight">FastAPI Python • Temporal Eye Tracker</p>
      </div>
    </div>
    <div class="flex items-center gap-2">
      <button id="btnLoadDemo" class="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-700 font-semibold rounded-md text-xs flex items-center gap-1.5 transition cursor-pointer shadow-sm">
        <i class="fa-solid fa-play text-[10px]"></i> Load Demo (satyam.mp4)
      </button>
      <button id="btnToggleWebcam" class="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 font-semibold rounded-md text-xs flex items-center gap-1.5 transition cursor-pointer border border-slate-700 shadow-sm">
        <i class="fa-solid fa-video text-[10px]"></i> <span id="webcamBtnText">Live Webcam</span>
      </button>
    </div>
  </header>

  <!-- 2. Small Top Cards Row (3 Small Sized Cards) -->
  <section class="grid grid-cols-1 md:grid-cols-3 gap-2 my-2 shrink-0">
    <!-- Small Card 1: Upload Video -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 flex flex-col justify-between shadow-sm">
      <div class="flex items-center justify-between text-xs font-semibold text-slate-300">
        <span class="flex items-center gap-1.5"><i class="fa-solid fa-cloud-arrow-up text-indigo-400 text-[11px]"></i> Video Input</span>
        <span id="statusMsg" class="text-[10px] hidden text-indigo-400 truncate max-w-[130px]"></span>
      </div>
      <label class="mt-1 flex items-center justify-center gap-2 px-2 py-1 bg-slate-950 hover:bg-slate-800/80 border border-dashed border-slate-700 hover:border-indigo-400 rounded cursor-pointer text-[11px] font-medium text-slate-300 transition">
        <i class="fa-regular fa-file-video text-indigo-400 text-xs"></i>
        <span id="uploadLabel">Upload MP4 Video</span>
        <input type="file" id="fileInput" accept="video/*" class="hidden" />
      </label>
    </div>

    <!-- Small Card 2: Current Alertness State -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 flex flex-col justify-between shadow-sm">
      <div class="flex items-center justify-between text-xs font-semibold text-slate-300">
        <span class="flex items-center gap-1.5"><i class="fa-solid fa-gauge-high text-emerald-400 text-[11px]"></i> Current State</span>
        <span class="text-[10px] text-slate-400">&lt;1s Awake • &gt;1s Drowsy • &gt;3s Sleep</span>
      </div>
      <div class="mt-1 flex items-center justify-center">
        <div id="liveBadge" class="px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-2 shadow badge-awake">
          <span id="liveDot" class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></span>
          <span id="liveStatusText">AWAKE (100%)</span>
        </div>
      </div>
    </div>

    <!-- Small Card 3: Video Summary Stats -->
    <div class="bg-slate-900/90 border border-slate-800 rounded-lg px-3 py-1.5 flex flex-col justify-between shadow-sm">
      <div class="flex items-center justify-between text-xs font-semibold text-slate-300">
        <span class="flex items-center gap-1.5"><i class="fa-solid fa-chart-pie text-cyan-400 text-[11px]"></i> Summary Stats</span>
      </div>
      <div id="summaryContent" class="mt-0.5 text-[11px] text-slate-400">
        <div class="flex items-center justify-between">
          <span class="text-slate-400">Awake: <strong class="text-emerald-400">100%</strong></span>
          <span class="text-slate-400">Drowsy: <strong class="text-amber-400">0%</strong></span>
          <span class="text-slate-400">Sleep: <strong class="text-rose-400">0%</strong></span>
        </div>
        <div class="w-full h-1.5 bg-slate-800 rounded-full flex overflow-hidden mt-1">
          <div style="width: 100%" class="bg-emerald-500"></div>
        </div>
      </div>
    </div>
  </section>

  <!-- 3. Main Split View: Left Corner Video Player & Opposite Right Side Alert Table (Zero-Scroll) -->
  <main class="flex-1 grid grid-cols-1 md:grid-cols-12 gap-2.5 min-h-0 overflow-hidden">

    <!-- LEFT CORNER: Live Playback with Alert Overlay (7 Cols) -->
    <div class="md:col-span-7 bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 flex flex-col min-h-0 overflow-hidden shadow-sm">
      <div class="flex items-center justify-between pb-1.5 mb-1.5 border-b border-slate-800 shrink-0 text-xs font-semibold text-slate-300">
        <span class="flex items-center gap-1.5">
          <i class="fa-solid fa-display text-indigo-400"></i> Live Playback with Alert Overlay
        </span>
        <span id="videoCaption" class="text-[10px] text-slate-400 hidden">Synced Face Bounding Box & Status Badge</span>
      </div>

      <!-- Video & Webcam Viewport -->
      <div class="flex-1 min-h-0 relative w-full h-full bg-black rounded border border-slate-800 overflow-hidden">
        <!-- Live Webcam View: Absolute full-bleed coverage -->
        <div id="webcamContainer" class="hidden absolute inset-0 w-full h-full overflow-hidden">
          <img id="webcamImg" src="" alt="Webcam Feed" style="width: 100%; height: 100%; object-fit: cover; display: block;" />
        </div>

        <!-- Video Player View: Absolute full-bleed coverage -->
        <div id="videoContainer" class="absolute inset-0 w-full h-full flex items-center justify-center overflow-hidden">
          <div id="emptyVideoPlaceholder" class="flex flex-col items-center justify-center text-slate-500">
            <i class="fa-solid fa-film text-3xl mb-1 text-slate-600"></i>
            <span class="text-xs">No video loaded. Click "Load Demo" or upload a file.</span>
          </div>
          <div id="videoPlayerWrapper" class="absolute inset-0 w-full h-full hidden overflow-hidden">
            <video id="videoElement" controls playsinline style="width: 100%; height: 100%; object-fit: cover; display: block;"></video>
            <canvas id="overlayCanvas" class="absolute top-0 left-0 w-full h-full pointer-events-none"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- OPPOSITE RIGHT SIDE: Alert History Table with Only Important Columns (5 Cols) -->
    <div class="md:col-span-5 bg-slate-900/90 border border-slate-800 rounded-lg p-2.5 flex flex-col min-h-0 overflow-hidden shadow-sm">
      <div class="flex items-center justify-between pb-1.5 mb-1.5 border-b border-slate-800 shrink-0">
        <span class="text-xs font-semibold text-slate-300 flex items-center gap-1.5">
          <i class="fa-solid fa-table-list text-emerald-400"></i> Alert History
        </span>
        <div class="flex items-center gap-1.5">
          <span class="text-[10px] text-slate-400">Filter:</span>
          <select id="statusFilter" class="bg-slate-950 border border-slate-700 text-[11px] rounded px-1.5 py-0.5 text-slate-200">
            <option value="all">All</option>
            <option value="awake">Awake</option>
            <option value="drowsy">Drowsy</option>
            <option value="sleeping">Sleep</option>
          </select>
        </div>
      </div>

      <!-- Scrollable Table Body Container (Only table scrolls internally, page has 0 scroll) -->
      <div class="flex-1 min-h-0 overflow-y-auto border border-slate-800 rounded bg-slate-950/40">
        <table class="w-full text-left text-xs text-slate-300">
          <thead class="text-[10px] uppercase bg-slate-950 text-slate-400 sticky top-0 border-b border-slate-800">
            <tr>
              <th class="px-2.5 py-1.5">Time</th>
              <th class="px-2.5 py-1.5">Status</th>
              <th class="px-2.5 py-1.5">Score</th>
              <th class="px-2.5 py-1.5">Duration</th>
            </tr>
          </thead>
          <tbody id="tableBody" class="divide-y divide-slate-800/60 font-mono text-xs">
            <tr>
              <td colspan="4" class="text-center py-10 text-slate-500 font-sans text-xs">
                No alert records available yet.
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>

  </main>

  <script>
    // State
    let detectionResults = [];
    let isWebcamOn = false;
    let webcamPollTimer = null;

    // Elements
    const fileInput = document.getElementById('fileInput');
    const uploadLabel = document.getElementById('uploadLabel');
    const btnLoadDemo = document.getElementById('btnLoadDemo');
    const btnToggleWebcam = document.getElementById('btnToggleWebcam');
    const webcamBtnText = document.getElementById('webcamBtnText');
    const webcamContainer = document.getElementById('webcamContainer');
    const webcamImg = document.getElementById('webcamImg');
    const videoContainer = document.getElementById('videoContainer');
    const emptyPlaceholder = document.getElementById('emptyVideoPlaceholder');
    const videoWrapper = document.getElementById('videoPlayerWrapper');
    const videoElem = document.getElementById('videoElement');
    const canvasElem = document.getElementById('overlayCanvas');
    const videoCaption = document.getElementById('videoCaption');
    const liveBadge = document.getElementById('liveBadge');
    const liveDot = document.getElementById('liveDot');
    const liveStatusText = document.getElementById('liveStatusText');
    const summaryContent = document.getElementById('summaryContent');
    const tableBody = document.getElementById('tableBody');
    const statusFilter = document.getElementById('statusFilter');
    const statusMsg = document.getElementById('statusMsg');

    function showStatus(msg, isError = false) {
      statusMsg.className = `text-[10px] truncate max-w-[150px] flex items-center gap-1 ${isError ? 'text-rose-400' : 'text-indigo-400'}`;
      statusMsg.innerHTML = isError ? `<i class="fa-solid fa-triangle-exclamation"></i> ${msg}` : `<i class="fa-solid fa-spinner fa-spin"></i> ${msg}`;
      statusMsg.classList.remove('hidden');
    }

    function hideStatus() {
      statusMsg.classList.add('hidden');
    }

    // Load Demo Video
    btnLoadDemo.addEventListener('click', () => {
      stopWebcam();
      showStatus("Analyzing demo...");
      uploadLabel.textContent = "satyam.mp4 (Demo)";
      loadVideoAndRunDetect("sample-demo.mp4", "/uploads/sample-demo.mp4");
    });

    // File Upload
    fileInput.addEventListener('change', async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      stopWebcam();
      uploadLabel.textContent = file.name;
      showStatus("Uploading " + file.name + "...");

      const formData = new FormData();
      formData.append("file", file);

      try {
        const res = await fetch("/upload", { method: "POST", body: formData });
        if (!res.ok) throw new Error("Upload failed");
        const data = await res.json();
        showStatus("Analyzing frames...");
        loadVideoAndRunDetect(data.video_id, data.video_url);
      } catch (err) {
        showStatus(err.message, true);
      }
    });

    async function loadVideoAndRunDetect(videoId, videoUrl) {
      webcamContainer.classList.add('hidden');
      videoContainer.classList.remove('hidden');
      emptyPlaceholder.classList.add('hidden');
      videoWrapper.classList.remove('hidden');
      videoCaption.classList.remove('hidden');

      videoElem.src = videoUrl;
      videoElem.load();

      try {
        const res = await fetch(`/detect/${videoId}`);
        if (!res.ok) throw new Error("Detection failed: HTTP " + res.status);
        const data = await res.json();
        hideStatus();

        detectionResults = data.results || [];
        renderSummary(data.summary);
        renderTable();
      } catch (err) {
        showStatus("Error: " + err.message, true);
      }
    }

    // Synchronize video player with canvas overlay
    videoElem.addEventListener('timeupdate', () => {
      if (detectionResults.length === 0) return;
      const t = videoElem.currentTime;

      let closest = detectionResults[0];
      let minDiff = Math.abs(closest.timestamp - t);

      for (let i = 1; i < detectionResults.length; i++) {
        const diff = Math.abs(detectionResults[i].timestamp - t);
        if (diff < minDiff) {
          minDiff = diff;
          closest = detectionResults[i];
        }
      }

      updateLiveStatus(closest.status, closest.score);
      drawOverlay(closest);
    });

    function resizeCanvas() {
      if (videoWrapper && videoWrapper.clientWidth > 0 && videoWrapper.clientHeight > 0) {
        canvasElem.width = videoWrapper.clientWidth;
        canvasElem.height = videoWrapper.clientHeight;
      }
    }

    videoElem.addEventListener('loadedmetadata', resizeCanvas);
    videoElem.addEventListener('play', resizeCanvas);
    window.addEventListener('resize', resizeCanvas);

    function drawOverlay(item) {
      const ctx = canvasElem.getContext('2d');
      ctx.clearRect(0, 0, canvasElem.width, canvasElem.height);

      if (!item || !item.bbox) return;

      const W = canvasElem.width;
      const H = canvasElem.height;
      const Vw = videoElem.videoWidth || 640;
      const Vh = videoElem.videoHeight || 360;

      // Mathematical mapping for object-cover fill
      const scale = Math.max(W / Vw, H / Vh);
      const offsetX = (W - Vw * scale) / 2;
      const offsetY = (H - Vh * scale) / 2;

      const [x, y, w, h] = item.bbox;
      const sx = x * scale + offsetX;
      const sy = y * scale + offsetY;
      const sw = w * scale;
      const sh = h * scale;

      let color = "#10b981"; // Awake = Green
      if (item.status === "drowsy") color = "#f59e0b"; // Yellow
      if (item.status === "sleeping") color = "#ef4444"; // Red

      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.strokeRect(sx, sy, sw, sh);

      const label = `${item.status.toUpperCase()} (${Math.round(item.score)}%)`;
      ctx.font = "bold 12px sans-serif";
      const tw = ctx.measureText(label).width;

      ctx.fillStyle = color;
      ctx.fillRect(sx, Math.max(0, sy - 22), tw + 12, 22);

      ctx.fillStyle = "#ffffff";
      ctx.fillText(label, sx + 6, Math.max(15, sy - 6));
    }

    function updateLiveStatus(status, score) {
      liveStatusText.textContent = `${status.toUpperCase()} (${Math.round(score)}%)`;
      liveBadge.className = `px-3 py-0.5 rounded-full text-xs font-bold uppercase tracking-wider flex items-center gap-2 shadow badge-${status}`;
      liveDot.className = `w-2 h-2 rounded-full animate-ping ${status === 'awake' ? 'bg-emerald-400' : (status === 'drowsy' ? 'bg-amber-400' : 'bg-rose-400')}`;
    }

    function renderSummary(summary) {
      if (!summary) return;
      summaryContent.innerHTML = `
        <div class="flex items-center justify-between text-[11px] mb-0.5">
          <span class="text-slate-400">Duration: <strong class="text-slate-200">${summary.video_duration_sec}s</strong></span>
          <span class="text-slate-400">Frames: <strong class="text-slate-200">${summary.total_samples}</strong></span>
        </div>
        <div class="w-full h-1.5 bg-slate-800 rounded-full flex overflow-hidden my-0.5">
          <div style="width: ${summary.awake_percent}%" class="bg-emerald-500"></div>
          <div style="width: ${summary.drowsy_percent}%" class="bg-amber-500"></div>
          <div style="width: ${summary.sleeping_percent}%" class="bg-rose-500"></div>
        </div>
        <div class="flex justify-between text-[10px] text-slate-400">
          <span class="text-emerald-400 font-medium">Awake ${summary.awake_percent}%</span>
          <span class="text-amber-400 font-medium">Drowsy ${summary.drowsy_percent}%</span>
          <span class="text-rose-400 font-medium">Sleep ${summary.sleeping_percent}%</span>
        </div>
      `;
    }

    // Render Alert History Table (Only 4 Important Columns: Time, Status, Score, Duration)
    function renderTable() {
      const filter = statusFilter.value;
      const rows = detectionResults.filter(r => filter === 'all' || r.status === filter);

      if (rows.length === 0) {
        tableBody.innerHTML = `<tr><td colspan="4" class="text-center py-10 text-slate-500 font-sans text-xs">No records matching filter.</td></tr>`;
        return;
      }

      tableBody.innerHTML = rows.map(r => `
        <tr class="hover:bg-slate-800/40 transition">
          <td class="px-2.5 py-1 text-slate-400 font-mono text-[11px]">${r.timestamp}s</td>
          <td class="px-2.5 py-1">
            <span class="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider badge-${r.status}">
              ${r.status}
            </span>
          </td>
          <td class="px-2.5 py-1 font-semibold text-xs text-slate-200">${Math.round(r.score)}%</td>
          <td class="px-2.5 py-1 text-[11px] text-slate-400 font-mono">${r.closed_duration}s</td>
        </tr>
      `).join('');
    }

    statusFilter.addEventListener('change', renderTable);

    // Webcam Toggle
    btnToggleWebcam.addEventListener('click', () => {
      if (isWebcamOn) {
        stopWebcam();
      } else {
        startWebcam();
      }
    });

    function startWebcam() {
      isWebcamOn = true;
      detectionResults = [];
      uploadLabel.textContent = "Live Webcam Feed";
      videoContainer.classList.add('hidden');
      webcamContainer.classList.remove('hidden');
      webcamImg.src = "/video_feed?" + Date.now();
      webcamBtnText.textContent = "Stop Webcam";
      btnToggleWebcam.classList.replace('bg-slate-800', 'bg-rose-600');
      btnToggleWebcam.classList.replace('hover:bg-slate-700', 'hover:bg-rose-700');

      if (webcamPollTimer) clearInterval(webcamPollTimer);
      webcamPollTimer = setInterval(async () => {
        try {
          const res = await fetch("/webcam_status");
          const data = await res.json();
          if (data.active && data.frame_count > 0) {
            updateLiveStatus(data.status, data.score);

            detectionResults.unshift({
              frame: data.frame_count,
              timestamp: data.timestamp,
              status: data.status,
              score: data.score,
              ear: data.ear,
              closed_duration: data.closed_duration
            });

            if (detectionResults.length > 200) {
              detectionResults.pop();
            }

            const total = detectionResults.length;
            const awakeCount = detectionResults.filter(r => r.status === 'awake').length;
            const drowsyCount = detectionResults.filter(r => r.status === 'drowsy').length;
            const sleepingCount = detectionResults.filter(r => r.status === 'sleeping').length;

            renderSummary({
              video_duration_sec: data.timestamp,
              total_samples: total,
              awake_percent: Math.round((awakeCount / total) * 100),
              drowsy_percent: Math.round((drowsyCount / total) * 100),
              sleeping_percent: Math.round((sleepingCount / total) * 100)
            });

            renderTable();
          }
        } catch (err) {
          console.error("Telemetry poll error:", err);
        }
      }, 400);
    }

    function stopWebcam() {
      isWebcamOn = false;
      if (webcamPollTimer) {
        clearInterval(webcamPollTimer);
        webcamPollTimer = null;
      }
      webcamContainer.classList.add('hidden');
      videoContainer.classList.remove('hidden');
      webcamImg.src = "";
      webcamBtnText.textContent = "Live Webcam";
      btnToggleWebcam.classList.replace('bg-rose-600', 'bg-slate-800');
      btnToggleWebcam.classList.replace('hover:bg-rose-700', 'hover:bg-slate-700');
    }
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import threading

    def open_browser():
        time.sleep(1.2)
        webbrowser.open("http://localhost:5000")

    print("\n" + "=" * 60)
    print(" 🚀 REAL-TIME ALERTNESS DETECTION DASHBOARD")
    print(" Starting server on: http://localhost:5000")
    print(" Automatically opening your browser...")
    print(" Press Ctrl+C in this terminal to stop.")
    print("=" * 60 + "\n")

    threading.Thread(target=open_browser, daemon=True).start()
    uvicorn.run("main:app", host="127.0.0.1", port=5000, reload=True)
