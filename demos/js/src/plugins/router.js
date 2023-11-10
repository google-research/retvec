/**
 * @fileoverview Description of this file.
 */
import {createRouter, createWebHistory} from "vue-router";

import Homepage from "../components/Homepage.vue";
import BinarizerDemo from "../components/BinarizerDemo.vue";
import EmotionDemo from "../components/EmotionDemo.vue";

// Load each demo on a separate route.
export default createRouter({
  history: createWebHistory("/retvec/"),
  routes: [
    {path: "/", name: "Home", component: Homepage},
    {path: "/binarizer_demo", name: "BinarizerDemo", component: BinarizerDemo},
    {path: "/emotion_demo", name: "EmotionDemo", component: EmotionDemo},
  ],
});
