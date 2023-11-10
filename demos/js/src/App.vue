<template>
  <v-app app>
    <v-app-bar-nav-icon @click="drawer = !drawer"></v-app-bar-nav-icon>
    <v-navigation-drawer
      v-model="drawer"
      rounded="xl"
      border="none"
      color="#f3f6fc"
      app
    >
      <v-sheet color="#f3f6fc" class="pl-4">
        <v-spacer></v-spacer>
        <v-list :items="items"></v-list>
      </v-sheet>
    </v-navigation-drawer>

    <v-main>
      <v-container class="py-8 px-6" fluid>
        <router-view></router-view>
      </v-container>
    </v-main>
  </v-app>
</template>

<script setup>
import {ref, watch, onMounted} from "vue";
import {useRoute} from "vue-router";

const route = useRoute();

const drawer = ref(true);

watch(
  () => route.params.id,
  () => updateDrawer(),
);

const updateDrawer = () => {
  console.log("hjere", route.fullPath);
  if (route.fullPath === "/") {
    drawer.value = false;
  }
};

const items = ref([
  {
    title: "Emotion demo",
    value: 1,
    props: {
      prependIcon: "mdi-emoticon-excited-outline",
      href: "/retvec/emotion_demo",
    },
  },
  {
    title: "Binarizer demo",
    value: 2,
    props: {
      prependIcon: "mdi-barcode",
      href: "/retvec/binarizer_demo",
    },
  },
]);

onMounted(() => updateDrawer());
</script>
