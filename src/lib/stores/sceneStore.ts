import { atom, onSet } from 'nanostores';
import { persistentAtom } from '@nanostores/persistent';

export { onSet };

// --- Light Blob State ---
// Controls the 3D light blob object
export const isBlobActive = atom(false);

// --- Model State ---
export const activeModel = persistentAtom<'light' | 'dark'>(
  'activeModel',
  'light'
);

// --- Theme State ---
// We use persistentAtom to save the user's choice
export const theme = persistentAtom<'light' | 'dark'>(
  'theme', // localStorage key
  'light' // Default value
);

// This runs on the client to keep the <html> tag in sync
onSet(theme, ({ newValue }) => {
  if (typeof document !== 'undefined') {
    const root = document.documentElement;
    root.classList.remove('light-mode', 'dark-mode');
    root.classList.add(newValue === 'dark' ? 'dark-mode' : 'light-mode');
    console.log(`Global Store: Theme set to ${newValue}`);
  }
});

// --- Layer State (Spatial) ---
export type LayerCoordinate = { x: number; y: number; z: number };

// Initial State: Front Layer (0, 0, 1) - "Topmost"
export const activeLayer = atom<LayerCoordinate>({ x: 0, y: 0, z: 1 });

// Valid Layers Map
// We define which coordinates actually have content
export const validLayers: LayerCoordinate[] = [
    { x: 0, y: 0, z: 1 }, // Front (Top) - The initial view
    { x: 0, y: 0, z: 0 }, // Back (Deep) - The immersive room
    { x: 1, y: 0, z: 0 }, // Right Side - Example of expansion
];

// Helper: Check if a coordinate exists in our map
export function isLayerValid(x: number, y: number, z: number): boolean {
    return validLayers.some(l => l.x === x && l.y === y && l.z === z);
}

// Helper: Get available moves from a position
export function getAvailableMoves(current: LayerCoordinate) {
    return {
        up: isLayerValid(current.x, current.y + 1, current.z),      // +Y
        down: isLayerValid(current.x, current.y - 1, current.z),    // -Y
        left: isLayerValid(current.x - 1, current.y, current.z),    // -X
        right: isLayerValid(current.x + 1, current.y, current.z),   // +X
        forward: isLayerValid(current.x, current.y, current.z - 1), // -Z (Deeper)
        backward: isLayerValid(current.x, current.y, current.z + 1) // +Z (Out)
    };
}
