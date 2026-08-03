import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'
import { viteStaticCopy } from 'vite-plugin-static-copy'

// COOP/COEP make the page cross-origin isolated. This is required for
// performance.measureUserAgentSpecificMemory() (the §1.1 peak-memory metric)
// and for onnxruntime-web multi-threaded WASM. Everything the app loads is
// same-origin (model + wasm are bundled), so require-corp is safe.
const crossOriginIsolation = {
  'Cross-Origin-Opener-Policy': 'same-origin',
  'Cross-Origin-Embedder-Policy': 'require-corp',
}

export default defineConfig({
  plugins: [
    vue(),
    // Serve the onnxruntime-web WASM binaries from our own origin (/ort/) so
    // no third-party CDN is contacted — keeps inference fully on-device.
    viteStaticCopy({
      targets: [
        { src: 'node_modules/onnxruntime-web/dist/*.wasm', dest: 'ort' },
        { src: 'node_modules/onnxruntime-web/dist/*.mjs', dest: 'ort' },
      ],
    }),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['model/best_model.onnx', 'model/best_model.int8.onnx', 'model/class_labels.json'],
      workbox: {
        // The FP32 model is ~17 MB; raise the precache size limit and cache the
        // model + wasm so the installed app works fully offline.
        maximumFileSizeToCacheInBytes: 30 * 1024 * 1024,
        globPatterns: ['**/*.{js,css,html,wasm,onnx,json}'],
      },
      manifest: {
        name: 'Tomato Leaf Disease Scanner',
        short_name: 'LeafScan',
        description: 'Offline, on-device tomato leaf disease detection.',
        theme_color: '#2e7d32',
        background_color: '#ffffff',
        display: 'standalone',
        start_url: '/',
      },
    }),
  ],
  server: { headers: crossOriginIsolation },
  preview: { headers: crossOriginIsolation },
  // onnxruntime-web ships prebuilt wasm; don't let Vite try to optimize it.
  optimizeDeps: { exclude: ['onnxruntime-web'] },
})
