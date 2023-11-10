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

<script>
export default {
  data: () => ({
    selectedItem: null,
    drawer: true,
    items: [
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
    ],
  }),
};
</script>
