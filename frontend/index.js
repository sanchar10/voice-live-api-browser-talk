/**
 * index.js — Browser client for the Voice Assistant.
 *
 * Audio pipeline:
 *   Mic → AudioWorklet (mic-processor) → PCM16 binary → WebSocket → server
 *   server → WebSocket binary (PCM16) → AudioWorklet (audio-processor) → Speaker
 *
 * Audio format in both directions: PCM16 little-endian, 24 kHz, mono.
 * This is the same format an iOS AVAudioEngine would produce, keeping the
 * WebSocket protocol client-agnostic.
 */

// ── State ──────────────────────────────────────────────────────────────────

let ws = null;
let audioContext = null;
let micStream = null;
let micSource = null;
let micWorklet = null;
let playbackWorklet = null;
let isTalking = false;

const SAMPLE_RATE = 24000;

// ── DOM refs ───────────────────────────────────────────────────────────────

const talkBtn       = document.getElementById("talkBtn");
const statusEl      = document.getElementById("status");
const transcriptEl  = document.getElementById("transcript");

// ── Toggle entry point ─────────────────────────────────────────────────────

function toggleTalking() {
    if (isTalking) {
        stopTalking();
    } else {
        startTalking();
    }
}

// ── Start ──────────────────────────────────────────────────────────────────

async function startTalking() {
    try {
        setStatus("Connecting…");

        // 1. Create AudioContext at 24 kHz (matches Voice Live native rate)
        audioContext = new AudioContext({ sampleRate: SAMPLE_RATE });

        // 2. Load AudioWorklet modules
        await audioContext.audioWorklet.addModule("/frontend/audio-processor.js");
        await audioContext.audioWorklet.addModule("/frontend/mic-processor.js");

        // 3. Set up playback worklet (speaker output)
        playbackWorklet = new AudioWorkletNode(audioContext, "audio-processor");
        playbackWorklet.connect(audioContext.destination);

        // 4. Open WebSocket
        const proto = location.protocol === "https:" ? "wss:" : "ws:";
        ws = new WebSocket(`${proto}//${location.host}/ws`);
        ws.binaryType = "arraybuffer";

        ws.onopen = async () => {
            setStatus("Connected — speak now");
            await startMicrophone();
            isTalking = true;
            talkBtn.textContent = "Stop Talking";
            talkBtn.classList.add("active");
        };

        ws.onmessage = (event) => {
            if (event.data instanceof ArrayBuffer) {
                // Binary = PCM16 audio from the agent → play it
                playAudio(event.data);
            } else {
                // Text = JSON event
                handleJsonEvent(JSON.parse(event.data));
            }
        };

        ws.onclose = () => {
            setStatus("Disconnected");
            cleanUp();
        };

        ws.onerror = () => {
            setStatus("Connection error");
            cleanUp();
        };

    } catch (err) {
        console.error(err);
        setStatus("Error: " + err.message);
        cleanUp();
    }
}

// ── Stop ───────────────────────────────────────────────────────────────────

function stopTalking() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.close();
    }
    cleanUp();
}

function cleanUp() {
    isTalking = false;
    talkBtn.textContent = "Start Talking";
    talkBtn.classList.remove("active");

    if (micStream) {
        micStream.getTracks().forEach((t) => t.stop());
        micStream = null;
    }
    if (audioContext) {
        audioContext.close().catch(() => {});
        audioContext = null;
    }
    micSource = null;
    micWorklet = null;
    playbackWorklet = null;
    ws = null;
}

// ── Microphone capture ─────────────────────────────────────────────────────

async function startMicrophone() {
    micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
            sampleRate: SAMPLE_RATE,
            channelCount: 1,
            echoCancellation: true,
            noiseSuppression: true,
        },
    });

    micSource = audioContext.createMediaStreamSource(micStream);
    micWorklet = new AudioWorkletNode(audioContext, "mic-processor");

    // mic-processor posts Float32 buffers → convert to PCM16 and send
    micWorklet.port.onmessage = (e) => {
        if (!ws || ws.readyState !== WebSocket.OPEN) return;
        const pcm16 = float32ToPcm16(e.data.pcm);
        ws.send(pcm16.buffer);
    };

    micSource.connect(micWorklet);
    // Don't connect micWorklet to destination (we don't want to hear ourselves)
}

// ── Playback ───────────────────────────────────────────────────────────────

function playAudio(arrayBuffer) {
    if (!playbackWorklet) return;
    const int16 = new Int16Array(arrayBuffer);
    const float32 = pcm16ToFloat32(int16);
    playbackWorklet.port.postMessage({ pcm: float32 });
}

// ── JSON events from server ────────────────────────────────────────────────

function handleJsonEvent(data) {
    switch (data.type) {
        case "transcript":
            addTranscript(data.role, data.text);
            break;
        case "mcp_status":
            addTranscript("mcp", data.text);
            break;
        case "speech_started":
            // User started speaking — clear the playback buffer (barge-in)
            if (playbackWorklet) {
                playbackWorklet.port.postMessage({ clear: true });
            }
            break;
        case "call_state":
            if (data.state === "ended") {
                setStatus("Session ended");
                cleanUp();
            }
            break;
    }
}

// ── Transcript display ─────────────────────────────────────────────────────

function addTranscript(role, text) {
    // Remove placeholder on first message
    const placeholder = transcriptEl.querySelector(".placeholder");
    if (placeholder) placeholder.remove();

    const div = document.createElement("div");
    div.className = `msg ${role}`;
    div.innerHTML = `<div class="role">${role}</div><div class="text">${escapeHtml(text)}</div>`;
    transcriptEl.appendChild(div);
    transcriptEl.scrollTop = transcriptEl.scrollHeight;
}

// ── Audio format conversion helpers ────────────────────────────────────────

function float32ToPcm16(float32Array) {
    const pcm16 = new Int16Array(float32Array.length);
    for (let i = 0; i < float32Array.length; i++) {
        const s = Math.max(-1, Math.min(1, float32Array[i]));
        pcm16[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return pcm16;
}

function pcm16ToFloat32(int16Array) {
    const float32 = new Float32Array(int16Array.length);
    for (let i = 0; i < int16Array.length; i++) {
        float32[i] = int16Array[i] / 0x8000;
    }
    return float32;
}

// ── Utilities ──────────────────────────────────────────────────────────────

function setStatus(msg) {
    statusEl.textContent = msg;
}

function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
}
