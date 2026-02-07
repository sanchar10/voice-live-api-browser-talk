/**
 * mic-processor.js — AudioWorklet for microphone capture.
 *
 * Accumulates input samples into a 4096-sample buffer, then posts
 * { pcm: Float32Array } to the main thread for WebSocket transmission.
 */
class MicProcessor extends AudioWorkletProcessor {
    constructor() {
        super();
        this.buffer = new Float32Array(4096);
        this.pos = 0;
    }

    process(inputs) {
        const input = inputs[0][0];
        if (!input) return true;

        for (let i = 0; i < input.length; i++) {
            this.buffer[this.pos++] = input[i];
            if (this.pos >= this.buffer.length) {
                this.port.postMessage({ pcm: this.buffer.slice() });
                this.pos = 0;
            }
        }
        return true;
    }
}
registerProcessor("mic-processor", MicProcessor);
