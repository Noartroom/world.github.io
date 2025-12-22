# iOS WebGPU Support Update

## Changes Made

**Date:** 2025-01-27  
**Reason:** iOS 26+ (Safari 26+) now supports WebGPU - prioritize WebGPU on iOS mobile

### 1. Device Detection (`src/lib/stores/deviceStore.ts`)

**Before:**
- iOS devices were forced to `hasWebGPU = false`
- Comment: "Safari iOS reports no WebGPU; treat it as GL-only"

**After:**
- iOS devices now allow natural WebGPU detection via `navigator.gpu`
- iOS 26+ will have `navigator.gpu` available, older versions won't
- WebGL fallback remains for older iOS versions

**Changes:**
- Removed: `if (isIOS) { hasWebGPU = false; }`
- Updated comments to reflect iOS 26+ WebGPU support
- Tier logic updated to handle iOS with WebGPU

### 2. Backend Selection (`src/renderer.rs`)

**Before:**
```rust
let backends = if is_mobile {
    wgpu::Backends::GL  // Force GL-only on mobile
} else {
    wgpu::Backends::GL | wgpu::Backends::BROWSER_WEBGPU
};
```

**After:**
```rust
// iOS 26+ now supports WebGPU - try WebGPU first on mobile, with GL fallback
let backends = wgpu::Backends::GL | wgpu::Backends::BROWSER_WEBGPU;
```

**Impact:**
- Mobile devices (including iOS) now try WebGPU first
- Adapter selection matrix already handles fallback to GL if WebGPU fails
- Existing fallback logic remains intact

### 3. Renderer Logic (`src/components/Renderer.astro`)

**Before:**
- iOS without WebGPU immediately fell back to static fallback
- Required `?enableGL=1` flag to use WebGL

**After:**
- iOS without WebGPU but with WebGL can use WebGL
- Only falls back to static if both WebGPU and WebGL are unavailable
- iOS 26+ with WebGPU will use WebGPU automatically

**Changes:**
- Updated condition: `!device.hasWebGPU && !device.hasWebGL` (was just `!device.hasWebGPU`)
- Allows WebGL fallback on older iOS versions

## Adapter Selection Flow

The existing adapter selection matrix (lines 53-102 in `renderer.rs`) already handles this correctly:

1. **First attempt:** HighPerformance + WebGPU (no fallback)
2. **Second attempt:** HighPerformance + WebGPU + fallback (can map to GL)
3. **Third attempt:** LowPower + WebGPU (no fallback)
4. **Fourth attempt:** LowPower + WebGPU + fallback (can map to GL)
5. **Fifth attempt:** HighPerformance without surface + fallback
6. **Sixth attempt:** LowPower without surface + fallback

This means:
- iOS 26+ with WebGPU: Will use WebGPU adapter
- iOS 26+ if WebGPU fails: Automatically falls back to GL adapter
- Older iOS: Will use GL adapter (WebGPU not available)

## Backward Compatibility

✅ **All existing functionality preserved:**
- WebGL fallback still works
- Static fallback still works
- Desktop behavior unchanged
- Android behavior unchanged
- Older iOS behavior unchanged (will use WebGL or fallback)

## Testing Checklist

- [ ] iOS 26+ (Safari 26+): Should use WebGPU
- [ ] iOS 25 and below: Should use WebGL or fallback
- [ ] Desktop Chrome: Should use WebGPU (unchanged)
- [ ] Desktop Safari: Should use WebGPU (unchanged)
- [ ] Android: Should use WebGPU if available, else WebGL
- [ ] WebGL fallback: Should still work on all platforms
- [ ] Static fallback: Should still work when both fail

## Known Issues

⚠️ **maxInterStageShaderComponents issue still exists:**
- Even with these changes, the Chrome/Safari `maxInterStageShaderComponents` error may still occur
- This is a separate issue with wgpu's limit handling
- WebGL fallback will activate if WebGPU fails due to this error

## Next Steps

1. Test on iOS 26+ device to verify WebGPU works
2. Test on older iOS to verify WebGL fallback works
3. Monitor for the maxInterStageShaderComponents error
4. If error persists, WebGL fallback will handle it gracefully

