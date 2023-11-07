import { createApp } from "vue";
import {createRouter, createWebHistory} from 'vue-router';

import App from "./App.vue";
import BinarizerDemo from "./components/BinarizerDemo.vue";
import EmotionDemo from "./components/EmotionDemo.vue";

// Load each demo on a separate route.
const router = createRouter( {
    history: createWebHistory(),
    routes: [
        { path: '/', name: "Default", redirect: { name: 'Home' }},
        {path: '/retvec/', name: 'Home', component: BinarizerDemo},
        {path: '/retvec/emotion_demo', name: 'EmotionDemo', component: EmotionDemo},
    ]
});

// Create the vue app.
createApp(App)
    .use(router)
    .mount("#app");