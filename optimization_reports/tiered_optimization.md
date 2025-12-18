current store is binary (isLowPower vs. Normal). To achieve the "SOTA vs. Safe" split without glitches, we need to distinguish between "Capable of Rendering" (WebGPU works) and "Capable of High-Fidelity" (Can handle 30MB assets + 4K textures).

Here is the improved implementation that hooks directly into your existing deviceStore.ts.

1. Refine deviceStore.ts (Add the "High End" Tier)

Your current isLowPower heuristic is excellent for downgrading (e.g. killing animations), but we need a specific flag for upgrading to SOTA assets.

We will add a tier to your state.

TypeScript
// deviceStore.ts
import { map } from 'nanostores';

export type DeviceState = {
  isMobile: boolean;
  hasWebGPU: boolean;
  isLowPower: boolean;
  prefersReducedMotion: boolean;
  isTouch: boolean;
  // NEW: Explicit Performance Tier
  tier: 'low' | 'balanced' | 'ultra'; 
};

export const deviceState = map<DeviceState>({
  isMobile: false,
  hasWebGPU: false,
  isLowPower: false,
  prefersReducedMotion: false,
  isTouch: false,
  tier: 'balanced' // Default to safe middle ground
});

export function initDeviceDetection() {
  if (typeof window === 'undefined') return;

  const ua = navigator.userAgent;
  const isMobile = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(ua);
  
  // @ts-ignore
  const hasWebGPU = !!navigator.gpu;
  const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;

  // --- REFINED HEURISTICS ---
  
  // 1. Default Baseline
  let isLowPower = !hasWebGPU; 
  let tier: 'low' | 'balanced' | 'ultra' = 'balanced';

  // @ts-ignore
  const cores = navigator.hardwareConcurrency || 4;
  // @ts-ignore
  const memory = navigator.deviceMemory || 4;
  // @ts-ignore
  const saveData = navigator.connection?.saveData === true;

  // 2. Identify Low Power (Aggressive Downgrade)
  if (saveData || cores < 4 || memory < 4) {
      isLowPower = true;
      tier = 'low';
  }

  // 3. Identify Ultra/SOTA Capable (Strict Upgrade)
  // Must have WebGPU, lots of RAM, NOT be on mobile data (heuristic), and NOT be "Low Power"
  if (hasWebGPU && !isLowPower && !isMobile && memory >= 8 && cores >= 6) {
      tier = 'ultra';
  }

  // Mobile SOTA Exception: High-end iPads/iPhones or flagship Androids
  if (isMobile && hasWebGPU && !isLowPower && memory >= 6) {
      // Mobile usually shouldn't download 30MB assets, so we keep them 'balanced' 
      // unless you specifically want SOTA on mobile. 
      // keeping 'balanced' forces the 2K model (Safe).
      tier = 'balanced'; 
  }

  deviceState.set({
    isMobile,
    hasWebGPU,
    isLowPower,
    prefersReducedMotion,
    isTouch,
    tier
  });

  console.log('Device Detection:', deviceState.get());
}
2. The Progressive Asset Pipeline

You still need to generate the Safe Base Model (2K textures) and keep your SOTA Model (Original).

Run these commands (using @gltf-transform/cli):

Bash
# 1. Generate SAFE Base Model (2K textures, WebP compression)
# This file will likely be ~3-5MB.
gltf-transform resize source_sota.glb model-base.glb --width 2048 --height 2048
gltf-transform webp model-base.glb model-base.glb --quality 80

# 2. Keep source_sota.glb as is (30MB+, 4K textures)
3. Renderer.astro (The Tiered Loader)

This is where we connect your Store to the Logic. We load "Safe" first, then check the Store to see if we are allowed to upgrade.

JavaScript
// Renderer.astro
import { deviceState, initDeviceDetection } from '../lib/stores/deviceStore';
// ... other imports

// --- Initialization ---
async function start(forceHighPerformance = false) {
    if (!canvas) return;

    // 1. Run Detection immediately
    initDeviceDetection();
    const device = deviceState.get();

    // 2. SAFETY CHECK: If Low Power, engage Fallback Mode immediately
    // (This uses your existing fallback logic)
    if (device.isLowPower && !forceHighPerformance) {
        activateFallbackMode();
        return;
    }

    // 3. LOAD BASE MODEL (The "Safe" 2K Version)
    // This renders quickly and guarantees no visual glitches or white screens.
    await init(); // Init WASM
    // Initialize Rust State
    state = await startRenderer(canvas, device.isMobile); 
    
    // Load the 2K model first
    await loadModel('base'); 

    // 4. PROGRESSIVE UPGRADE (The SOTA Check)
    // We check the Tier computed in your store + Network conditions
    await attemptSotaUpgrade(device);

    // ... [Observers and Event Listeners setup] ...
}

async function attemptSotaUpgrade(device) {
    // 1. Check Store Tier
    if (device.tier !== 'ultra') {
        console.log(`Rendering Tier: ${device.tier.toUpperCase()} (Sticking to Base Model)`);
        return;
    }

    // 2. Check Network (Dynamic check, separate from static hardware capabilities)
    // @ts-ignore
    const connection = navigator.connection;
    if (connection) {
        // Don't download 30MB on 3G or if RTT is high
        if (connection.saveData || connection.effectiveType !== '4g') {
            console.log("Tier is ULTRA, but network is slow. Skipping SOTA upgrade.");
            return;
        }
    }

    console.log("🚀 Tier ULTRA detected. Fetching SOTA assets...");

    try {
        // 3. Lazy Load SOTA Model
        const response = await fetch('/models/source_sota.glb');
        if (!response.ok) throw new Error('Network error');
        
        const bytes = new Uint8Array(await response.arrayBuffer());

        // 4. Hot-Swap in Rust
        if (state) {
            state.loadModelFromBytes(bytes); // Rust hot-swaps geometry/textures
            console.log("✨ Upgraded to SOTA Resolution");
        }
    } catch (e) {
        console.warn("SOTA upgrade failed, safely stayed on Base model.", e);
    }
}

// Updated loadModel helper
const loadModel = async (modelType) => {
    // Map abstract names to files
    let path = '/models/model-base.glb'; // Default Safe 2K
    
    if (modelType === 'dark') path = '/models/model-dark-base.glb'; 
    // Note: If you have a dark theme SOTA, logic gets complex. 
    // Start with just upgrading the main light model for now.

    // ... [Your existing fetch/load logic]
};
4. Preventing Glitches on Resize

With nanostores, we can react to changes. If a user resizes their desktop window to be very small, we don't necessarily want to downgrade the model (re-downloading is bad), but we do want to cap the DPI to save heat.

Add this logic inside your existing triggerResize function in Renderer.astro:

JavaScript
// Renderer.astro - triggerResize()

function triggerResize() {
    if (!container || !canvas) return;
    
    const device = deviceState.get(); // Read from your store

    // 1. Smart DPI Cap
    // If Tier is Ultra, we allow 2.0 (Retina).
    // If Balanced/Mobile, we cap at 1.5.
    // If Low, we cap at 1.0.
    let cap = 1.0;
    if (device.tier === 'ultra') cap = 2.0;
    else if (device.tier === 'balanced') cap = 1.5;

    // 2. Window Size Safety
    // Even on Ultra Desktop, if window is small, don't render 2.0 DPI
    if (window.innerWidth < 800) cap = Math.min(cap, 1.5);

    // Apply
    const pixelRatio = window.devicePixelRatio || 1;
    currentDpr = Math.min(targetDpr, Math.min(pixelRatio, cap));

    // ... [Rest of resize logic]
}