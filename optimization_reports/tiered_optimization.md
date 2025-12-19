1. The Asset Pipeline (Execute First)

Goal: Create the "Safe" (Base) and "Ultra" (SOTA) assets without using Draco.

Run these commands in your terminal:

Bash
# 1. Base Model (Balanced Tier)
# Target: ~3-5MB. Safe for all devices.
# 2K textures + WebP compression + Resize
gltf-transform resize source_sota.glb model-base.glb --width 2048 --height 2048
gltf-transform webp model-base.glb model-base.glb --quality 80

# 2. SOTA Model (Ultra Tier)
# Target: 30MB+. Desktop Only.
# Keep your original file.
cp source_sota.glb model-sota.glb
2. The Thermal-Aware Device Store

File: src/lib/stores/deviceStore.ts Change: Added cleanup logic for the battery listener to prevent memory leaks.

TypeScript
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
  isLowPower: boolean;
  isTouch: boolean;
  tier: Tier;
  networkTier: 'high' | 'low';
};

export const deviceState = map<DeviceState>({
  isMobile: false,
  hasWebGPU: false,
  isLowPower: false,
  isTouch: false,
  tier: 'balanced',
  networkTier: 'high'
});

let batteryCleanup: (() => void) | null = null;

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
  const isTouch = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
  
  const cores = navigator.hardwareConcurrency || 4;
  const memory = navigator.deviceMemory || 4; 
  
  const conn = navigator.connection;
  const isSlowNetwork = conn ? (conn.saveData || conn.effectiveType === '2g' || conn.effectiveType === '3g') : false;
  
  // --- TIER LOGIC ---
  let tier: Tier = 'balanced';
  let isLowPower = false;

  // 1. LOW TIER (Fail-safe)
  if (!hasWebGPU || isSlowNetwork || (isMobile && memory < 4)) {
    tier = 'low';
    isLowPower = true;
  }
  // 2. ULTRA TIER (SOTA Desktop)
  // Strict: Must be Desktop (No Touch), WebGPU, >8GB RAM, High Cores.
  // CRITICAL: !isTouch protects iPad Pro M2 from overheating.
  else if (hasWebGPU && !isMobile && !isTouch && memory >= 8 && cores >= 6) {
    tier = 'ultra';
  }
  // 3. BALANCED TIER (Default)
  else {
    tier = 'balanced';
  }

  // Battery Listener (Async)
  if (navigator.getBattery) {
    try {
      const battery = await navigator.getBattery();
      const checkBattery = () => {
          if (!battery.charging && battery.level < 0.2) {
             // Force downgrade if low battery
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
    isLowPower,
    isTouch,
    tier,
    networkTier: isSlowNetwork ? 'low' : 'high'
  });
}
3. The Memory-Safe Rust Hot-Swap

File: src/lib.rs Change: Explicitly dropping the old model prevents the "Out of Memory" crash.

Rust
#[wasm_bindgen]
impl State {
    // ... existing methods ...

    // REPLACES: load_model_from_bytes
    #[wasm_bindgen(js_name = "loadModelFromBytes")]
    pub fn load_model_from_bytes(&mut self, bytes: &[u8]) {
        // 1. SAFETY: Explicitly drop the old model first.
        // This ensures RAM is freed BEFORE we parse the new 30MB file.
        if self.game.model.is_some() {
            self.game.model = None; // Explicit drop
        }

        // 2. Parse & Load New Model
        // This will cause the ~100-300ms freeze your friend mentioned.
        // We will mask this with the spinner in Astro.
        match load_model_from_bytes(
            &self.renderer.device, 
            &self.renderer.queue, 
            bytes, 
            &self.renderer.material_layout,
            &self.renderer.mipmap_pipeline_linear,
            &self.renderer.mipmap_pipeline_srgb,
            &self.renderer.mipmap_bind_group_layout,
            self.game.tx.clone()
        ) {
            Ok(model) => {
                self.game.model_center = model.center;
                self.game.model_extent = model.extent;
                self.game.model = Some(model);
                web_sys::console::log_1(&"Model Hot-Swap Complete".into());
            },
            Err(e) => {
                web_sys::console::error_1(&format!("Failed to parse GLB: {}", e).into());
            }
        }
    }
}
4. The "Zero-Glitch" Loader

File: src/components/Renderer.astro Change: Implements the "Debounce Delay" and uses the UI Spinner to mask the hot-swap freeze.

JavaScript
import { deviceState, initDeviceDetection } from '../lib/stores/deviceStore';
import { activeModel } from '../lib/stores/sceneStore';

// ... imports (init, startRenderer, etc.)

// DOM Elements
const loader = document.getElementById('loader');

async function start(forceHighPerformance = false) {
    if (!canvas) return;

    // 1. Detect Tier
    await initDeviceDetection();
    const device = deviceState.get();

    // 2. Fallback for Potato Devices
    if (device.tier === 'low' && !forceHighPerformance) {
        activateFallbackMode();
        return;
    }

    // 3. Init Engine
    if (loader) loader.classList.remove('hidden');
    await init();
    state = await startRenderer(canvas, device.isMobile);

    // 4. LOAD SAFE MODEL (Balanced/Base)
    // Always load this first. Fast TTI.
    await loadModel('base');
    
    // Hide loader after base model is ready
    if (loader) loader.classList.add('hidden');

    // 5. PROGRESSIVE UPGRADE (The "Ultra" Path)
    if (device.tier === 'ultra') {
        // Wait 2 seconds for main thread to settle
        console.log("⏳ Tier Ultra detected. Scheduling SOTA upgrade...");
        setTimeout(() => triggerSotaUpgrade(), 2000);
    }
}

async function triggerSotaUpgrade() {
    const device = deviceState.get();
    
    if (device.networkTier === 'low') return;

    console.log("🚀 Triggering SOTA Asset Upgrade...");
    
    try {
        // A. Show "Buffering" or subtle loader (Optional)
        // We re-show the loader text to explain the micro-freeze
        if (loader) {
            loader.textContent = "UPGRADING TEXTURES...";
            loader.classList.remove('hidden');
        }

        // B. Fetch SOTA model (30MB+)
        // This happens in background, no freeze yet
        const response = await fetch('/models/model-sota.glb');
        if (!response.ok) throw new Error('Download failed');
        
        const bytes = new Uint8Array(await response.arrayBuffer());

        // C. Hot-Swap (The Freeze happens here)
        // We use requestAnimationFrame to ensure the Loader is rendered 
        // BEFORE we freeze the thread with Rust.
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                if (state) {
                    state.loadModelFromBytes(bytes);
                    console.log("✨ Visuals upgraded to 4K SOTA");
                    
                    // Hide loader again
                    if (loader) loader.classList.add('hidden');
                }
            });
        });

    } catch (e) {
        console.warn("SOTA upgrade failed, staying on Base model.", e);
        if (loader) loader.classList.add('hidden');
    }
}

const loadModel = async (quality = 'base') => {
    // quality is 'base' (2K) or 'sota' (4K)
    // Note: Always default to 'base' for initial load logic
    const path = quality === 'sota' ? '/models/model-sota.glb' : '/models/model-base.glb';
    
    try {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const bytes = new Uint8Array(await response.arrayBuffer());
        
        if (state) state.loadModelFromBytes(bytes);
    } catch(e) {
        console.error("Model load failed", e);
    }
};

The Fix: Enabling "Legacy 3D" (WebGL)

We need to make two changes:

Rust: Tell wgpu to explicitly allow the OpenGL (WebGL) backend.

TypeScript: Update detection to allow "WebGL-only" devices into the "Low" or "Balanced" tier instead of kicking them to the fallback image.

1. Update renderer.rs (Enable WebGL Support)

Your current code uses Instance::default(), which on the web often strictly prefers WebGPU. We need to explicitly request Backends::all() (or specifically GL + WebGPU) to ensure it hunts for a WebGL2 context when WebGPU is missing. 

Modify renderer.rs:

Rust
pub async fn new(canvas: HtmlCanvasElement, is_mobile: bool, sample_count: u32) -> Result<Self, JsValue> {
    // OLD: let instance = wgpu::Instance::default();
    
    // NEW: Explicitly allow WebGL (GL) and WebGPU (BROWSER_WEBGPU)
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::GL | wgpu::Backends::BROWSER_WEBGPU,
        ..Default::default()
    });

    // ... rest of your code
Note: You already have logic in renderer.rs to disable MSAA on the GL backend (if info.backend == wgpu::Backend::Gl), so this will integrate perfectly. 

2. Update deviceStore.ts (Allow WebGL Detection)

Currently, isLowPower defaults to true if !hasWebGPU. We need to change this to only fallback if both WebGPU and WebGL are missing. 

Modify initDeviceDetection in deviceStore.ts:

TypeScript
// Add this helper to check for WebGL2 support
function hasWebGLSupport() {
  try {
    const canvas = document.createElement('canvas');
    return !!(window.WebGL2RenderingContext && canvas.getContext('webgl2'));
  } catch (e) {
    return false;
  }
}

export function initDeviceDetection() {
  if (typeof window === 'undefined') return;

  // ... [Existing detection logic] ...
  
  // @ts-ignore
  const hasWebGPU = !!navigator.gpu;
  const hasWebGL = hasWebGLSupport(); // NEW CHECK

  // ... [Heuristics] ...

  // --- REFINED TIER LOGIC ---
  let tier: Tier = 'balanced';
  let isLowPower = false;

  // 1. FALLBACK (Static Image)
  // Only if BOTH WebGPU and WebGL are missing, or device is extremely weak
  if ((!hasWebGPU && !hasWebGL) || isSlowNetwork || (isMobile && memory < 2)) {
      tier = 'low';
      isLowPower = true; // Triggers static fallback
  }
  
  // 2. LEGACY 3D (Safari / DDG / Ecosia)
  // Has WebGL but no WebGPU. We force them to "Low" or "Balanced" tier.
  else if (!hasWebGPU && hasWebGL) {
      console.log("WebGL Legacy Mode Detected (Safari/WebView)");
      tier = 'balanced'; // Renders 2K model
      isLowPower = false; // ALLOWS 3D RENDERING
  }

  // 3. ULTRA TIER (SOTA Desktop)
  // Must have WebGPU specifically
  else if (hasWebGPU && !isMobile && !isTouch && memory >= 8 && cores >= 6) {
    tier = 'ultra';
  }
  
  // 4. BALANCED TIER (Default WebGPU Mobile/Laptop)
  else {
    tier = 'balanced';
  }

  // ... [Rest of function]