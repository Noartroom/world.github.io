import { atom, onSet } from 'nanostores';
import { persistentAtom } from '@nanostores/persistent';

export { onSet };

// --- Dynamic Light State ---
// Controls cursor-directed 3D lighting (enabled by default)
export const isDynamicLightActive = atom(true);

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