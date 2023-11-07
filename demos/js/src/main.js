import { createApp } from "vue";
import {createRouter, createWebHistory} from 'vue-router';

import App from "./App.vue";
import BinarizerDemo from "./components/BinarizerDemo.vue";

// Load each demo on a separate route.
const router = createRouter( {
    history: createWebHistory(),
    routes: [
        {path: '/', name: 'Home', component: BinarizerDemo},
    ]
});

// Create the vue app.
createApp(App)
    .use(router)
    .mount("#app");