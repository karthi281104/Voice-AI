/**
 * VoiceDeskAudioProcessor — AudioWorklet for Real-Time Microphone Streaming
 * Resamples incoming microphone audio from native browser sampleRate (44.1 kHz / 48 kHz)
 * to standard 16 kHz mono PCM frames (320 samples = 20 ms), suitable for Silero VAD & Whisper.
 */
class VoiceDeskAudioProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.targetSampleRate = 16000;
    this.frameSize = 320; // 20 ms at 16 kHz
    this.buffer = [];
    this.resamplePhase = 0;
  }

  process(inputs, outputs, parameters) {
    const inputChannel = inputs[0]?.[0];
    if (!inputChannel || inputChannel.length === 0) {
      return true;
    }

    const nativeRate = sampleRate; // Global AudioWorkletContext sampleRate (e.g. 44100 or 48000)

    if (nativeRate === this.targetSampleRate) {
      // Direct pass-through if browser already supports 16 kHz context
      for (let i = 0; i < inputChannel.length; i++) {
        this.buffer.push(inputChannel[i]);
      }
    } else {
      // Linear interpolation downsampler to 16 kHz
      const ratio = nativeRate / this.targetSampleRate;
      let pos = this.resamplePhase;

      while (pos < inputChannel.length) {
        const index = Math.floor(pos);
        const nextIndex = Math.min(index + 1, inputChannel.length - 1);
        const frac = pos - index;
        const sample = inputChannel[index] * (1 - frac) + inputChannel[nextIndex] * frac;
        this.buffer.push(sample);
        pos += ratio;
      }
      this.resamplePhase = pos - inputChannel.length;
    }

    // Emit packed 16-bit PCM frames (320 samples = 640 bytes)
    while (this.buffer.length >= this.frameSize) {
      const frame = this.buffer.splice(0, this.frameSize);
      const pcm = new Int16Array(this.frameSize);
      for (let i = 0; i < this.frameSize; i++) {
        const s = Math.max(-1, Math.min(1, frame[i]));
        pcm[i] = s < 0 ? s * 32768 : s * 32767;
      }
      this.port.postMessage(pcm.buffer, [pcm.buffer]);
    }

    return true;
  }
}

registerProcessor("voicedesk-audio-processor", VoiceDeskAudioProcessor);
