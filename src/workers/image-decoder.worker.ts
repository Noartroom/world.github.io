// Web Worker for off-thread image decoding
// This prevents UI stutter when loading large textures

self.onmessage = async function(e: MessageEvent) {
    const { imageData, mimeType, imageIndex, textureType } = e.data;
    
    try {
        // Create blob from image data
        const blob = new Blob([imageData], { type: mimeType });
        
        // Decode image in worker using OffscreenCanvas (if available) or ImageBitmap
        let imageBitmap: ImageBitmap;
        
        if (typeof OffscreenCanvas !== 'undefined' && 'createImageBitmap' in self) {
            // Use createImageBitmap directly (most efficient)
            imageBitmap = await createImageBitmap(blob);
        } else {
            // Fallback: create ImageBitmap from blob
            imageBitmap = await createImageBitmap(blob);
        }
        
        // Transfer ImageBitmap back to main thread (zero-copy transfer)
        // Note: ImageBitmap is transferable, so we use transferList
        self.postMessage({
            success: true,
            imageIndex,
            textureType,
            imageBitmap,
        }, [imageBitmap]);
    } catch (error) {
        // Send error back to main thread
        self.postMessage({
            success: false,
            imageIndex,
            textureType,
            error: error instanceof Error ? error.message : String(error),
        });
    }
};

