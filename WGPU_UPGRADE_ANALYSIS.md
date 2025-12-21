# WGPU Upgrade Analysis

## Current State

**Current wgpu version:** 0.20.0  
**Issue:** `maxInterStageShaderComponents` limit not recognized by Chrome/Safari WebGPU implementation

## Problem Analysis

The error `"The limit 'maxInterStageShaderComponents' with a non-undefined value is not recognized"` indicates that:
1. Chrome/Safari's WebGPU implementation doesn't support this limit
2. The limit has been deprecated in favor of `maxInterStageShaderVariables` (Chrome 133+)
3. wgpu 0.20.0 may be using deprecated WebGPU limits

## Available wgpu Versions

**IMPORTANT:** wgpu has changed versioning scheme!

- **0.20.1** (current) - Released ~2024, has maxInterStageShaderComponents issue
- **0.21.x** - Potential fixes for WebGPU compatibility
- **0.22.x** - Further improvements  
- **0.23.x** - Continued improvements
- **...**
- **27.0.1** (latest as of 2025) - Major version jump indicates significant changes

**⚠️ CRITICAL:** The jump from 0.20.1 to 27.0.1 suggests:
- Major API changes likely
- Possible breaking changes in core APIs
- Version numbering scheme change (may have skipped versions or changed format)

## Dependency Compatibility Matrix

### Current Dependencies
```
wgpu = "0.20.0"
web-sys = "0.3.69"
wasm-bindgen = "0.2.92"
wasm-bindgen-futures = "0.4.42"
glam = "0.28.0"
gltf = "1.4.1"
bytemuck = "1.16.0"
flume = "0.11.0"
```

### Compatibility Concerns

1. **web-sys 0.3.69** - May need update for newer wgpu versions
2. **wasm-bindgen 0.2.92** - Should be compatible, but newer versions available
3. **glam 0.28.0** - Should be fine, widely compatible
4. **gltf 1.4.1** - Should be fine
5. **bytemuck 1.16.0** - Should be fine

## WGPU API Usage Analysis

### Core APIs Used (from codebase scan):

1. **Instance & Adapter:**
   - `wgpu::Instance::new()`
   - `wgpu::InstanceDescriptor`
   - `wgpu::RequestAdapterOptions`
   - `adapter.request_device()`
   - `adapter.limits()`

2. **Surface:**
   - `wgpu::Surface<'static>`
   - `wgpu::SurfaceTarget::Canvas()`
   - `wgpu::SurfaceConfiguration`
   - `surface.get_capabilities()`
   - `surface.configure()`

3. **Device & Queue:**
   - `wgpu::Device`
   - `wgpu::Queue`
   - `wgpu::DeviceDescriptor`
   - `wgpu::Limits::downlevel_webgl2_defaults()`

4. **Resources:**
   - `wgpu::Texture`, `wgpu::TextureView`
   - `wgpu::Buffer`
   - `wgpu::BindGroup`, `wgpu::BindGroupLayout`
   - `wgpu::RenderPipeline`
   - `wgpu::ShaderModule`

5. **Rendering:**
   - `wgpu::RenderPassDescriptor`
   - `wgpu::CommandEncoder`
   - `wgpu::util::DeviceExt` (for `create_buffer_init`)

### Potential Breaking Changes to Check:

1. **Limits API:**
   - `maxInterStageShaderComponents` → `maxInterStageShaderVariables` (deprecated)
   - `downlevel_webgl2_defaults()` may have changed

2. **Surface API:**
   - `SurfaceTarget` may have changed
   - `SurfaceConfiguration` fields may differ

3. **Device Creation:**
   - `DeviceDescriptor` structure may have changed
   - Error types may differ

4. **Backend Flags:**
   - `wgpu::Backends::BROWSER_WEBGPU` may have changed name
   - `wgpu::Backend::Gl` enum value

## Recommended Upgrade Path

### ⚠️ WARNING: Major Version Jump Detected

The latest wgpu version is **27.0.1**, which is a massive jump from **0.20.1**. This suggests:
- Major API restructuring
- Significant breaking changes
- Possible versioning scheme change

### Option 1: Conservative Incremental Update (RECOMMENDED)
1. **First:** Update to **wgpu 0.21.0**
   - Test for breaking changes
   - Fix compilation errors
   - Verify WebGPU compatibility
   - Check if maxInterStageShaderComponents issue is resolved

2. **Then:** Update to **wgpu 0.22.0** if 0.21 doesn't fix the issue
   - Repeat testing process

3. **Continue:** Incrementally update through minor versions
   - Stop if maxInterStageShaderComponents issue is resolved
   - Don't jump to 27.0.1 without testing intermediate versions

### Option 2: Check Latest Stable 0.x Version
1. Check what the latest **0.x** version is (may be 0.23, 0.24, etc.)
2. Update to that version if it's reasonable
3. Avoid jumping to 27.0.1 without understanding the changes

### Option 3: Direct Update to Latest (HIGH RISK)
1. Update directly to **wgpu 27.0.1**
2. Expect significant breaking changes
3. May require major code refactoring
4. Only recommended if intermediate versions don't fix the issue

## Required Code Changes (Expected)

### 1. Limits Handling
```rust
// OLD (0.20.0):
let mut required_limits = wgpu::Limits::downlevel_webgl2_defaults();

// NEW (0.21+):
// May need to use adapter.limits() directly or new limit construction
// maxInterStageShaderComponents removed, use maxInterStageShaderVariables
```

### 2. Device Descriptor
```rust
// May need to check if DeviceDescriptor fields changed
// Especially: required_limits, required_features structure
```

### 3. Error Handling
```rust
// Error types may have changed
// .map_err() calls may need adjustment
```

### 4. Surface Configuration
```rust
// SurfaceConfiguration may have new/removed fields
// Check: view_formats, desired_maximum_frame_latency, etc.
```

## Testing Checklist

After upgrade, test:
- [ ] Code compiles without errors
- [ ] Chrome desktop renders 3D models
- [ ] Safari mobile renders 3D models
- [ ] Firefox renders 3D models (regression test)
- [ ] WebGL fallback still works
- [ ] No console errors about WebGPU limits
- [ ] Performance is maintained
- [ ] WASM bundle size hasn't increased significantly

## Risk Assessment

**Low Risk:**
- glam, bytemuck, flume updates (if needed)
- Minor API changes in wgpu

**Medium Risk:**
- Limits API changes
- Device descriptor changes
- Surface configuration changes

**High Risk:**
- Breaking changes in core rendering pipeline
- Backend enum changes
- Major API restructuring

## Recommended Action Plan

1. **Create a backup branch** before making changes
2. **Update wgpu incrementally** (0.20 → 0.21 → 0.22 → 0.23)
3. **Fix compilation errors** at each step
4. **Test thoroughly** after each update
5. **Update related dependencies** if required by wgpu
6. **Document any breaking changes** encountered

## Alternative Solution (If Upgrade Fails)

If upgrading wgpu causes too many breaking changes, consider:
1. **Patching wgpu locally** to remove maxInterStageShaderComponents
2. **Using a fork** of wgpu with the fix
3. **Waiting for wgpu patch release** that fixes the issue
4. **Using adapter limits directly** without modification (already attempted)

## Summary & Recommendations

### Current Situation
- **Current Version:** wgpu 0.20.1
- **Latest Version:** wgpu 27.0.1 (major version jump!)
- **Issue:** `maxInterStageShaderComponents` not recognized by Chrome/Safari
- **Code Status:** Reverted to original working state

### Recommended Approach

**DO NOT upgrade to 27.0.1 directly** - the version jump is too large and likely has breaking changes.

**Recommended Strategy:**
1. **First, try incremental updates:**
   - Update to wgpu 0.21.0
   - Test if maxInterStageShaderComponents issue is resolved
   - If not, continue to 0.22.0, 0.23.0, etc.
   - Stop when the issue is fixed OR when you reach a reasonable version

2. **Before upgrading, check:**
   - wgpu changelog for versions 0.21+ to see if maxInterStageShaderComponents was addressed
   - GitHub issues for wgpu related to this specific error
   - Whether newer 0.x versions fix the Chrome/Safari compatibility

3. **If incremental updates don't work:**
   - Consider if the issue is actually in wgpu or in how we're using it
   - May need to wait for a wgpu patch release
   - Or use a workaround (like using adapter limits directly, which we tried)

### Risk Level: **HIGH**

Upgrading wgpu carries significant risk due to:
- Large version gap (0.20.1 → 27.0.1)
- Unknown breaking changes
- Potential need for major code refactoring
- Dependency compatibility issues

### Alternative Solutions to Consider First

1. **Check if wgpu 0.20.x has a patch release** that fixes this
2. **Use adapter limits directly** (we tried this but it didn't work)
3. **Check wgpu GitHub for known issues** and workarounds
4. **Consider if the issue is browser-specific** and may resolve with browser updates

## Next Steps

1. ✅ **Code reverted** - Original working code restored
2. ✅ **Research completed** - Found that wgpu versioning changed (0.20.x → 26.x+)
3. ✅ **Upgrade attempted** - Updated to wgpu 26.0.6
4. ✅ **Compilation successful** - Code compiles without errors
5. ⏳ **Testing required** - Need to test in Chrome/Safari to verify maxInterStageShaderComponents fix
6. ⏳ **Document API changes** - If any breaking changes found during testing

## Upgrade Status: IN PROGRESS

**Current Status:**
- ✅ Updated wgpu from 0.20.0 to 26.0.6
- ✅ Code compiles successfully
- ⏳ Testing in browsers required to verify fix

**Next Actions:**
1. Build WASM bundle and test in Chrome desktop
2. Test in Safari mobile
3. Verify no maxInterStageShaderComponents errors
4. If successful, document the fix
5. If issues persist, try wgpu 27.0.4 or 28.0.0

