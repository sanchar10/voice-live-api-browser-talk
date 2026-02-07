/**
 * audio-processor.js — AudioWorklet for speaker playback.
 *
 * Receives Float32 PCM chunks via port.postMessage({ pcm: Float32Array })
 * and feeds them to the audio output at a steady 128-sample cadence.
 * Handles { clear: true } to flush the buffer on barge-in.
 */
class AudioProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(0);
        this.port.onmessage = (e) => {
            if (e.data.clear) {
                this.buffer = new Float32Array(0);
                return;
            }
            // Append incoming PCM to ring buffer
            const combined = new Float32Array(this.buffer.length + e.data.pcm.length);
            combined.set(this.buffer);
            combined.set(e.data.pcm, this.buffer.length);
            this.buffer = combined;
        };
    }

    process(_inputs, outputs) {
        const out = outputs[0][0];
        if (this.buffer.length >= out.length) {
            out.set(this.buffer.subarray(0, out.length));
            this.buffer = this.buffer.subarray(out.length);
        } else {
            out.fill(0);
        }
        return true;
    }
}
registerProcessor("audio-processor", AudioProcessor);
