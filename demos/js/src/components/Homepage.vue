<template>
  <div class="text-h2 pb-3">RetVec demos</div>

  <p>
    RETVec is a state-of-the-art text vectorizer which works directly on text
    inputs to create resilient classification models. Models trained with RETVec
    achieve better classification performance with fewer parameters and exhibit
    stronger resilience against adversarial attacks and typos, as reported
    <a href="https://arxiv.org/abs/2302.09207"> in our paper</a>.
  </p>

  <v-row align="center" justify="center" class="mt-3">
    <v-col cols="auto" v-for="(item, i) in ITEMS" :key="i">
      <v-card
        class="mx-auto"
        max-width="320"
        :title="item.title"
        :prepend-icon="item.prependIcon"
      >
        <v-card-text>{{ item.description }}</v-card-text>
        <v-card-actions>
          <router-link :to="item.href">
            <v-btn>Start</v-btn>
          </router-link>
        </v-card-actions>
      </v-card>
    </v-col>
  </v-row>
</template>

<script setup>
const ITEMS = [
  {
    title: "Emotion classifier demo",
    prependIcon: "mdi-emoticon-excited-outline",
    href: "/emotion_demo",
    description:
      "See how RetVec can make an emotion classifier more robust to typos and l33t speak while inproving its performance.",
  },
  {
    title: "Binarizer demo",
    prependIcon: "mdi-barcode",
    href: "/binarizer_demo",
    description: "See how RetVec encodes words into a bit array.",
  },
];
</script>
