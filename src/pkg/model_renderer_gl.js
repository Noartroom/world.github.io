// GL-only shim: reuse the main renderer entry points.
// If you produce a dedicated GL build, replace these imports with that bundle.
import init, { startRenderer as startRendererGL } from './model_renderer.js';

export default init;
export { startRendererGL };

