import { atom } from 'nanostores';

export const isAudioContextReady = atom(false);

let audioContext: AudioContext | null = null;
let analyser: AnalyserNode | null = null; // Main (Mono mix) analyser
let analyserL: AnalyserNode | null = null; // Left Channel
let analyserR: AnalyserNode | null = null; // Right Channel
let splitter: ChannelSplitterNode | null = null;

let dataArray: Uint8Array<ArrayBuffer> | null = null;
let dataArrayL: Uint8Array<ArrayBuffer> | null = null;
let dataArrayR: Uint8Array<ArrayBuffer> | null = null;

// To avoid re-connecting the same element multiple times
const connectedElements = new WeakSet<HTMLMediaElement>();

export function initAudioContext() {
  if (typeof window === 'undefined') return null;
  
  if (!audioContext) {
    const AudioCtor = window.AudioContext || (window as any).webkitAudioContext;
    if (!AudioCtor) {
        console.error("Web Audio API is not supported in this browser");
        return null;
    }
    audioContext = new AudioCtor();
    
    // Main Analyser (Mono/Mix)
    analyser = audioContext.createAnalyser();
    analyser.fftSize = 512;
    analyser.smoothingTimeConstant = 0.8;
    dataArray = new Uint8Array(analyser.frequencyBinCount);

    // Stereo Analysers
    analyserL = audioContext.createAnalyser();
    analyserL.fftSize = 512;
    analyserL.smoothingTimeConstant = 0.8;
    dataArrayL = new Uint8Array(analyserL.frequencyBinCount);

    analyserR = audioContext.createAnalyser();
    analyserR.fftSize = 512;
    analyserR.smoothingTimeConstant = 0.8;
    dataArrayR = new Uint8Array(analyserR.frequencyBinCount);

    // Splitter
    splitter = audioContext.createChannelSplitter(2);
    splitter.connect(analyserL, 0); // Connect Output 0 (Left) to analyserL
    splitter.connect(analyserR, 1); // Connect Output 1 (Right) to analyserR

    isAudioContextReady.set(true);
    console.log("AudioContext initialized (Stereo Mode)");
  }
  return audioContext;
}

export function connectAudioElement(element: HTMLMediaElement) {
  const ctx = initAudioContext();
  if (!ctx || !analyser || !splitter) return;

  if (connectedElements.has(element)) {
      return;
  }

  try {
      if (!element.crossOrigin) {
         element.crossOrigin = "anonymous";
      }

      const source = ctx.createMediaElementSource(element);
      
      // 1. Connect to Main Analyser (for general visualizer)
      source.connect(analyser);
      
      // 2. Connect to Splitter (for Stereo analysis)
      source.connect(splitter);

      // 3. Connect to Output (Speakers)
      analyser.connect(ctx.destination);
      
      connectedElements.add(element);
      console.log(`✅ Audio element connected (Stereo): ${element.id}`);
      
      element.onplay = () => {
          console.log(`▶️ Audio element playing: ${element.id}`);
          resumeAudioContext();
      };
      
  } catch (e) {
      console.warn("❌ Error connecting audio element:", e);
  }
}

export function getAudioData(): Uint8Array<ArrayBuffer> {
    if (!analyser || !dataArray) return new Uint8Array(0);
    analyser.getByteFrequencyData(dataArray);
    return dataArray;
}

export function getStereoData() {
    if (!analyserL || !analyserR || !dataArrayL || !dataArrayR) {
        return { left: 0, right: 0 };
    }
    
    analyserL.getByteFrequencyData(dataArrayL);
    analyserR.getByteFrequencyData(dataArrayR);

    const sumL = dataArrayL.reduce((a, b) => a + b, 0);
    const sumR = dataArrayR.reduce((a, b) => a + b, 0);
    
    const avgL = sumL / dataArrayL.length;
    const avgR = sumR / dataArrayR.length;

    return { left: avgL, right: avgR };
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

// Frequency band analysis (3-band: Bass/Mid/Treble)
// FFT size 512 = 256 frequency bins
// Sample rate typically 44100 or 48000 Hz
// Each bin represents: sampleRate / (fftSize / 2) Hz
export function getFrequencyBands(sampleRate: number = 48000): { bass: number; mid: number; treble: number } {
    if (!analyser || !dataArray) {
        return { bass: 0, mid: 0, treble: 0 };
    }
    
    analyser.getByteFrequencyData(dataArray);
    
    const fftSize = analyser.fftSize;
    const binCount = analyser.frequencyBinCount; // Usually 256 for fftSize 512
    const binWidth = sampleRate / fftSize; // Hz per bin
    
    // Frequency ranges
    // Bass: 20-250 Hz
    // Mid: 250-4000 Hz
    // Treble: 4000-20000 Hz
    const bassEnd = Math.floor(250 / binWidth);
    const midStart = Math.floor(250 / binWidth);
    const midEnd = Math.floor(4000 / binWidth);
    const trebleStart = Math.floor(4000 / binWidth);
    
    let bassSum = 0;
    let midSum = 0;
    let trebleSum = 0;
    let bassCount = 0;
    let midCount = 0;
    let trebleCount = 0;
    
    for (let i = 0; i < binCount; i++) {
        const value = dataArray[i];
        if (i < bassEnd) {
            bassSum += value;
            bassCount++;
        } else if (i >= midStart && i < midEnd) {
            midSum += value;
            midCount++;
        } else if (i >= trebleStart) {
            trebleSum += value;
            trebleCount++;
        }
    }
    
    return {
        bass: bassCount > 0 ? bassSum / bassCount : 0,
        mid: midCount > 0 ? midSum / midCount : 0,
        treble: trebleCount > 0 ? trebleSum / trebleCount : 0
    };
}

// Spectral Centroid (brightness) - weighted average frequency
export function getSpectralCentroid(sampleRate: number = 48000): number {
    if (!analyser || !dataArray) {
        return 0;
    }
    
    analyser.getByteFrequencyData(dataArray);
    
    const fftSize = analyser.fftSize;
    const binCount = analyser.frequencyBinCount;
    const binWidth = sampleRate / fftSize;
    
    let weightedSum = 0;
    let magnitudeSum = 0;
    
    for (let i = 0; i < binCount; i++) {
        const magnitude = dataArray[i];
        const frequency = i * binWidth;
        weightedSum += frequency * magnitude;
        magnitudeSum += magnitude;
    }
    
    return magnitudeSum > 0 ? weightedSum / magnitudeSum : 0;
}

// Spectral Rolloff - frequency below which 85% of energy is contained
export function getSpectralRolloff(sampleRate: number = 48000, threshold: number = 0.85): number {
    if (!analyser || !dataArray) {
        return 0;
    }
    
    analyser.getByteFrequencyData(dataArray);
    
    const fftSize = analyser.fftSize;
    const binCount = analyser.frequencyBinCount;
    const binWidth = sampleRate / fftSize;
    
    let totalEnergy = 0;
    for (let i = 0; i < binCount; i++) {
        totalEnergy += dataArray[i];
    }
    
    const targetEnergy = totalEnergy * threshold;
    let cumulativeEnergy = 0;
    
    for (let i = 0; i < binCount; i++) {
        cumulativeEnergy += dataArray[i];
        if (cumulativeEnergy >= targetEnergy) {
            return i * binWidth;
        }
    }
    
    return (binCount - 1) * binWidth;
}

// Zero Crossing Rate (noisiness) - requires time domain data
export function getZeroCrossingRate(): number {
    if (!analyser || !dataArray) {
        return 0;
    }
    
    // Get time domain data for zero crossing analysis
    const timeData = new Uint8Array(analyser.fftSize);
    analyser.getByteTimeDomainData(timeData);
    
    let crossings = 0;
    for (let i = 1; i < timeData.length; i++) {
        const prev = timeData[i - 1] - 128; // Center around 0
        const curr = timeData[i] - 128;
        if ((prev >= 0 && curr < 0) || (prev < 0 && curr >= 0)) {
            crossings++;
        }
    }
    
    return crossings / timeData.length;
}

// Spatial Audio Signals
export function getSpatialSignals(): {
    stereoWidth: number;
    phaseDifference: number;
    channelSeparation: number;
    azimuth: number;
} {
    if (!analyserL || !analyserR || !dataArrayL || !dataArrayR) {
        return { stereoWidth: 0, phaseDifference: 0, channelSeparation: 0, azimuth: 0 };
    }
    
    analyserL.getByteFrequencyData(dataArrayL);
    analyserR.getByteFrequencyData(dataArrayR);
    
    // Stereo Width: Correlation between L/R channels
    let correlation = 0;
    let lSum = 0;
    let rSum = 0;
    let lSqSum = 0;
    let rSqSum = 0;
    
    for (let i = 0; i < dataArrayL.length; i++) {
        const l = dataArrayL[i];
        const r = dataArrayR[i];
        correlation += l * r;
        lSum += l;
        rSum += r;
        lSqSum += l * l;
        rSqSum += r * r;
    }
    
    const lMean = lSum / dataArrayL.length;
    const rMean = rSum / dataArrayL.length;
    const lStd = Math.sqrt((lSqSum / dataArrayL.length) - (lMean * lMean));
    const rStd = Math.sqrt((rSqSum / dataArrayL.length) - (rMean * rMean));
    
    const stereoWidth = (lStd > 0 && rStd > 0) 
        ? (correlation / dataArrayL.length - lMean * rMean) / (lStd * rStd)
        : 0;
    
    // Phase Difference: Average phase difference between channels
    // Simplified: use amplitude difference as proxy
    const phaseDifference = Math.abs(lMean - rMean) / 255.0;
    
    // Channel Separation: How different the channels are
    let separation = 0;
    for (let i = 0; i < dataArrayL.length; i++) {
        const diff = Math.abs(dataArrayL[i] - dataArrayR[i]);
        separation += diff;
    }
    const channelSeparation = separation / (dataArrayL.length * 255.0);
    
    // Azimuth estimation: -1 (left) to +1 (right) based on energy difference
    const totalL = lSum;
    const totalR = rSum;
    const total = totalL + totalR;
    const azimuth = total > 0 ? (totalR - totalL) / total : 0;
    
    return {
        stereoWidth: Math.max(-1, Math.min(1, stereoWidth)), // Clamp to [-1, 1]
        phaseDifference: Math.max(0, Math.min(1, phaseDifference)),
        channelSeparation: Math.max(0, Math.min(1, channelSeparation)),
        azimuth: Math.max(-1, Math.min(1, azimuth))
    };
}

// RMS Energy per frequency band
export function getRMSEnergyPerBand(sampleRate: number = 48000): { bass: number; mid: number; treble: number } {
    if (!analyser || !dataArray) {
        return { bass: 0, mid: 0, treble: 0 };
    }
    
    analyser.getByteFrequencyData(dataArray);
    
    const fftSize = analyser.fftSize;
    const binCount = analyser.frequencyBinCount;
    const binWidth = sampleRate / fftSize;
    
    const bassEnd = Math.floor(250 / binWidth);
    const midStart = Math.floor(250 / binWidth);
    const midEnd = Math.floor(4000 / binWidth);
    const trebleStart = Math.floor(4000 / binWidth);
    
    let bassSqSum = 0;
    let midSqSum = 0;
    let trebleSqSum = 0;
    let bassCount = 0;
    let midCount = 0;
    let trebleCount = 0;
    
    for (let i = 0; i < binCount; i++) {
        const value = dataArray[i] / 255.0; // Normalize to 0-1
        const sqValue = value * value;
        
        if (i < bassEnd) {
            bassSqSum += sqValue;
            bassCount++;
        } else if (i >= midStart && i < midEnd) {
            midSqSum += sqValue;
            midCount++;
        } else if (i >= trebleStart) {
            trebleSqSum += sqValue;
            trebleCount++;
        }
    }
    
    return {
        bass: bassCount > 0 ? Math.sqrt(bassSqSum / bassCount) : 0,
        mid: midCount > 0 ? Math.sqrt(midSqSum / midCount) : 0,
        treble: trebleCount > 0 ? Math.sqrt(trebleSqSum / trebleCount) : 0
    };
}

// Peak detection - find the dominant frequency
export function getPeakFrequency(sampleRate: number = 48000): number {
    if (!analyser || !dataArray) {
        return 0;
    }
    
    analyser.getByteFrequencyData(dataArray);
    
    const fftSize = analyser.fftSize;
    const binCount = analyser.frequencyBinCount;
    const binWidth = sampleRate / fftSize;
    
    let maxValue = 0;
    let peakBin = 0;
    
    for (let i = 0; i < binCount; i++) {
        if (dataArray[i] > maxValue) {
            maxValue = dataArray[i];
            peakBin = i;
        }
    }
    
    return peakBin * binWidth;
}

// Comprehensive audio analysis - returns all signals
export function getComprehensiveAudioAnalysis(sampleRate: number = 48000) {
    const frequencyBands = getFrequencyBands(sampleRate);
    const rmsEnergy = getRMSEnergyPerBand(sampleRate);
    const spatial = getSpatialSignals();
    
    return {
        // Frequency analysis
        frequencyBands,
        rmsEnergy,
        spectralCentroid: getSpectralCentroid(sampleRate),
        spectralRolloff: getSpectralRolloff(sampleRate),
        peakFrequency: getPeakFrequency(sampleRate),
        zeroCrossingRate: getZeroCrossingRate(),
        
        // Spatial analysis
        spatial,
        
        // Basic signals
        intensity: getAverageVolume() / 255.0,
        stereo: getStereoData()
    };
}

export function resumeAudioContext() {
    if (audioContext && audioContext.state === 'suspended') {
        audioContext.resume();
    }
}