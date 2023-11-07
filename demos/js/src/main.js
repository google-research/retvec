import { createApp } from "vue";
import {createRouter, createWebHistory} from 'vue-router';

import App from "./App.vue";
import Homepage from "./components/Homepage.vue";
import BinarizerDemo from "./components/BinarizerDemo.vue";
import EmotionDemo from "./components/EmotionDemo.vue";

// Load each demo on a separate route.
const router = createRouter( {
    history: createWebHistory('/retvec/'),
    routes: [
        {path: '/', name: 'Home', component: Homepage},
        {path: '/binarizer_demo', name: 'BinarizerDemo', component: BinarizerDemo},
        {path: '/emotion_demo', name: 'EmotionDemo', component: EmotionDemo},
    ]
});

// Create the vue app.
createApp(App)
    .use(router)
    .mount("#app");