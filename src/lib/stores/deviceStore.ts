import { map } from 'nanostores';

export type DeviceState = {
  isMobile: boolean;
  hasWebGPU: boolean;
  isLowPower: boolean;
  prefersReducedMotion: boolean;
  isTouch: boolean;
};

export const deviceState = map<DeviceState>({
  isMobile: false,
  hasWebGPU: false,
  isLowPower: false, // Default to false, detect later
  prefersReducedMotion: false,
  isTouch: false
});

export function initDeviceDetection() {
  if (typeof window === 'undefined') return;

  const ua = navigator.userAgent;
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
  
  // @ts-ignore - navigator.gpu might not be in types
  const hasWebGPU = !!navigator.gpu;
  
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

  // Heuristics for "Low Power" / Low Compute
  // 1. Explicit Low Power Mode (if detectable, usually via Battery API - experimental)
  // 2. Low logical processors (< 4)
  // 3. Low device memory (if available, < 4GB)
  // 4. No WebGPU support (Automatic fallback)
  // 5. Data Saver / Lite Mode active
  
  let isLowPower = !hasWebGPU; // If no WebGPU, we MUST be in "fallback" mode basically

  // @ts-ignore
  if (navigator.connection && navigator.connection.saveData) {
      isLowPower = true;
      console.log('Low Power Mode: Data Saver active');
  }

  if (hasWebGPU && !isLowPower) {
      // @ts-ignore
      const cores = navigator.hardwareConcurrency || 4;
      // @ts-ignore
      const memory = navigator.deviceMemory || 8; // Default to 8 if not supported

      if (isMobile) {
          // Stricter on mobile
          if (cores < 6 || memory < 4) {
              isLowPower = true;
          }
      } else {
          // Desktop
          if (cores < 4) {
              isLowPower = true;
          }
      }
  }

  deviceState.set({
    isMobile,
    hasWebGPU,
    isLowPower,
    prefersReducedMotion,
    isTouch
  });

  console.log('Device Detection:', deviceState.get());
}
