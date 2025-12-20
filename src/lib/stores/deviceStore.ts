import { map } from 'nanostores';

interface NetworkInformation {
  saveData?: boolean;
  effectiveType?: 'slow-2g' | '2g' | '3g' | '4g';
}

interface BatteryManager extends EventTarget {
  level: number;
  charging: boolean;
}

declare global {
  interface Navigator {
    gpu?: any;
    connection?: NetworkInformation;
    deviceMemory?: number;
    getBattery?: () => Promise<BatteryManager>;
  }
}

export type Tier = 'low' | 'balanced' | 'ultra';

export type DeviceState = {
  isMobile: boolean;
  hasWebGPU: boolean;
  hasWebGL: boolean;
  isLowPower: boolean;
  isTouch: boolean;
  tier: Tier;
  networkTier: 'high' | 'low';
};

export const deviceState = map<DeviceState>({
  isMobile: false,
  hasWebGPU: false,
  hasWebGL: false,
  isLowPower: false,
  isTouch: false,
  tier: 'balanced',
  networkTier: 'high'
});

let batteryCleanup: (() => void) | null = null;

function hasWebGLSupport() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2'));
  } catch (e) {
    return false;
  }
}

export async function initDeviceDetection() {
  if (typeof window === 'undefined') return;

  // Cleanup previous listeners if re-initialized
  if (batteryCleanup) {
    batteryCleanup();
    batteryCleanup = null;
  }

  const ua = navigator.userAgent;
  const isAndroid = /Android/i.test(ua);
  // iPadOS 13+ detection (MacIntel + TouchPoints)
  const isIOS = /iPhone|iPad|iPod/i.test(ua) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
  const isMobile = isAndroid || isIOS || /webOS|BlackBerry|IEMobile|Opera Mini/i.test(ua);

  // @ts-ignore
  const hasWebGPU = !!navigator.gpu;
  const hasWebGL = hasWebGLSupport();
  const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  
  const cores = navigator.hardwareConcurrency || 4;
  const memory = navigator.deviceMemory || 4; 
  
  const conn = navigator.connection;
  const isSlowNetwork = conn ? (conn.saveData || conn.effectiveType === '2g' || conn.effectiveType === '3g') : false;
  
  // --- TIER LOGIC ---
  let tier: Tier = 'balanced';
  let isLowPower = false;

  // 1. FALLBACK (Static Image)
  if ((!hasWebGPU && !hasWebGL) || isSlowNetwork || (isMobile && memory < 2)) {
    tier = 'low';
    isLowPower = true;
  }
  // iOS Safari: if WebGPU is absent, WebGL paths are often unstable — prefer fallback
  else if (!hasWebGPU && isIOS) {
    tier = 'low';
    isLowPower = true;
  }
  // 2. LEGACY 3D (Safari / WebView)
  else if (!hasWebGPU && hasWebGL) {
    console.log('WebGL Legacy Mode Detected (Safari/WebView)');
    tier = 'balanced';
    isLowPower = false;
  }
  // 3. ULTRA TIER (Desktop SOTA)
  else if (hasWebGPU && !isMobile && !isTouch && memory >= 8 && cores >= 6) {
    tier = 'ultra';
  }
  // 4. BALANCED TIER (Default)
  else {
    tier = 'balanced';
  }

  // Battery Listener (Async)
  if (navigator.getBattery) {
    try {
      const battery = await navigator.getBattery();
      const checkBattery = () => {
        if (!battery.charging && battery.level < 0.2) {
          const current = deviceState.get();
          if (current.tier === 'ultra') deviceState.setKey('tier', 'balanced');
          deviceState.setKey('isLowPower', true);
        }
      };
      
      checkBattery();
      battery.addEventListener('levelchange', checkBattery);
      
      // Save cleanup function
      batteryCleanup = () => {
        battery.removeEventListener('levelchange', checkBattery);
      };
    } catch (e) { /* Ignore */ }
  }

  deviceState.set({
    isMobile,
    hasWebGPU,
    hasWebGL,
    isLowPower,
    isTouch,
    tier,
    networkTier: isSlowNetwork ? 'low' : 'high'
  });
}
