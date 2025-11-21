import { atom } from 'nanostores';

export const isAudioContextReady = atom(false);

let audioContext: AudioContext | null = null;
let analyser: AnalyserNode | null = null;
let dataArray: Uint8Array | null = null;

// To avoid re-connecting the same element multiple times
const connectedElements = new WeakSet<HTMLAudioElement>();

export function initAudioContext() {
  if (typeof window === 'undefined') return null;
  
  if (!audioContext) {
    const AudioCtor = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtor) {
        console.error("Web Audio API is not supported in this browser");
        return null;
    }
    audioContext = new AudioCtor();
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512; // Trade-off between resolution and speed
    analyser.smoothingTimeConstant = 0.8;
    
    const bufferLength = analyser.frequencyBinCount;
    dataArray = new Uint8Array(bufferLength);
    
    isAudioContextReady.set(true);
    console.log("AudioContext initialized");
  }
  return audioContext;
}

export function connectAudioElement(element: HTMLAudioElement) {
  const ctx = initAudioContext();
  if (!ctx || !analyser) return;

  if (connectedElements.has(element)) {
      // Already connected, skip
      return;
  }

  try {
      const source = ctx.createMediaElementSource(element);
      source.connect(analyser);
      analyser.connect(ctx.destination);
      connectedElements.add(element);
      console.log("Audio element connected to analyser:", element.id);
  } catch (e) {
      console.warn("Error connecting audio element:", e);
  }
}

export function getAudioData(): Uint8Array {
    if (!analyser || !dataArray) return new Uint8Array(0);
    analyser.getByteFrequencyData(dataArray);
    return dataArray;
}

export function getAverageVolume(): number {
    const data = getAudioData();
    if (data.length === 0) return 0;
    
    let sum = 0;
    for (let i = 0; i < data.length; i++) {
        sum += data[i];
    }
    return sum / data.length;
}

export function resumeAudioContext() {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
}