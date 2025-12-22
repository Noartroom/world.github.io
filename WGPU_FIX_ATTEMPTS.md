# WGPU maxInterStageShaderComponents Fix Attempts

## Problem
Chrome/Safari WebGPU implementation doesn't recognize `maxInterStageShaderComponents` limit when set to a non-undefined value. This causes device creation to fail.

## Attempts Made

### Attempt 1: Use adapter limits directly
- **Approach:** Use `adapter.limits()` directly without modification
- **Result:** ❌ Failed - wgpu still includes maxInterStageShaderComponents in the request
- **Version:** wgpu 26.0.6

### Attempt 2: Use Limits::default()
- **Approach:** Use `wgpu::Limits::default()` with minimal overrides
- **Result:** ❌ Failed - default limits still include the problematic field
- **Version:** wgpu 28.0.0

### Attempt 3: Set maxInterStageShaderComponents to 0
- **Approach:** Clone adapter limits and set `max_inter_stage_shader_components = 0`
- **Result:** ⏳ Testing - Setting to 0 may cause wgpu to omit it or handle it differently
- **Version:** wgpu 28.0.0

## Current Status

**Latest Fix (Attempt 3):**
- Using wgpu 28.0.0 (latest)
- Cloning adapter limits
- Setting `max_inter_stage_shader_components = 0`
- Mobile: WebGPU enabled on iOS 26+ (as requested)

## Root Cause Analysis

The issue appears to be that wgpu internally serializes the Limits struct to WebGPU's format, and even when we don't explicitly set `maxInterStageShaderComponents`, wgpu may be including it based on:
1. The Limits struct having a default value
2. wgpu's internal serialization logic
3. The adapter reporting this limit (even though Chrome doesn't recognize it)

## Next Steps if Current Fix Fails

1. **Check wgpu source code** - See how Limits are serialized to WebGPU
2. **Try wgpu git version** - Use latest from GitHub (may have unreleased fix)
3. **File wgpu issue** - Report this as a bug if it's not already known
4. **Consider workaround** - Force WebGL fallback on Chrome until wgpu fixes this
5. **Check Chrome version** - Ensure using latest Chrome that should support WebGPU properly

## Mobile WebGPU Support

✅ **Updated:** Mobile now tries WebGPU first (iOS 26+ support)
- Backends: `GL | BROWSER_WEBGPU` for all platforms
- WebGL fallback still available if WebGPU fails

