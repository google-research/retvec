import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],

  build: {
    rollupOptions: {
      output: {
        // Do not add hashes to the url of asset files.
        // This is important as TFJS loads the weights of the model
        // based on its url.
        assetFileNames: `assets/[name].[ext]`,
      },
    },
  },
});
