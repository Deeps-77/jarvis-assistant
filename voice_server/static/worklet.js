// Mic capture AudioWorklet: forwards mono Float32 frames to the main thread.
// The AudioContext is created with sampleRate 16000, so no resampling needed.
class MicCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.active = true;
    this.port.onmessage = (e) => {
      if (e.data && e.data.cmd === 'stop') this.active = false;
      if (e.data && e.data.cmd === 'start') this.active = true;
    };
  }

  process(inputs) {
    if (this.active && inputs.length > 0 && inputs[0].length > 0) {
      // Copy: the underlying buffer is reused after return.
      this.port.postMessage(inputs[0][0].slice(0));
    }
    return true;
  }
}

registerProcessor('mic-capture', MicCapture);
