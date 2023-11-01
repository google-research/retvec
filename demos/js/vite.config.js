import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],

  build: {
    rollupOptions: {
      external: ['@tensorflow/tfjs/dist/tf.fesm.js'],
    },
  },
});
