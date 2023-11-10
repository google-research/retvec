<template>
  <Demo title="How are you feeling?" :snackbar="message">
    <div class="pb-2">
      <p>
        In this demo, we showcase a model that classifiers
        {{ LABELS.length }} types of emotion exhibited in a piece of text. This
        model is trained on the
        <a
          href="https://ai.googleblog.com/2021/10/goemotions-dataset-for-fine-grained.html"
          >Go Emotions dataset</a
        >.
      </p>
      <p>
        To understand how this model has been trained using Retvec, please take
        a look at our Colab
        <a
          href="https://colab.research.google.com/github/google-research/retvec/blob/main/notebooks/train_retvec_model_tf.ipynb"
        >
          <img
            src="https://colab.research.google.com/assets/colab-badge.svg"
            alt="Open In Colab"
          /> </a
        >. To see how we use RetVecJS to run your model in your browser, please
        <a
          href="https://github.com/google-research/retvec/blob/main/demos/js/src/components/EmotionDemo.vue"
        >
          read the code </a
        >, then edit it on Github Codespaces
        <a href="https://codespaces.new/google-research/retvec"
          ><img
            src="https://github.com/codespaces/badge.svg"
            alt="Open in GitHub Codespaces"
            style="
              max-width: 100%;
              width: 11rem;
              transform: translateY(20%);
            " /></a
        >.
      </p>
    </div>

    <div class="text-h4 pt-3 pb-5">Inputs</div>

    <v-text-field
      clearable
      label="Text to classify"
      placeholder="Write some text"
      v-model="userText"
      append-inner-icon="mdi-arrow-right"
      @click:append-inner="runInference"
    >
    </v-text-field>

    <v-chip-group>
      <span class="mr-2"> Or choose from an example:</span>
      <v-chip
        size="x-small"
        variant="outlined"
        rounded
        v-for="example in EXAMPLES"
        :text="example"
        @click="userText = example"
      ></v-chip>
    </v-chip-group>

    <div v-if="topScores">
      <div class="text-h4 pt-3 pb-5">Verdict</div>
      <BarsVisualization
        :labels="topLabels"
        :scores="topScores"
      ></BarsVisualization>
    </div>
  </Demo>
</template>

<script setup>
/**
 * Emotion classification demo.
 *
 * - To see this run:
 *    https://google-research.github.io/retvec/emotion_demo
 *
 * - To start in on your local machine or on Codespaces:
 *    https://github.com/google-research/retvec/blob/main/demos/js/README.md
 */

import {onMounted, ref, watch} from "vue";
import debounce from "lodash/debounce";
import Demo from "@/components/Demo.vue";
import BarsVisualization from "@/components/BarsVisualization.vue";

import * as tf from "@tensorflow/tfjs";
import RetVec from "../../retvecjs/retvec.ts";

const LABELS = [
  "admiration",
  "amusement",
  "anger",
  "annoyance",
  "approval",
  "caring",
  "confusion",
  "curiosity",
  "desire",
  "disappointment",
  "disapproval",
  "disgust",
  "embarrassment",
  "excitement",
  "fear",
  "gratitude",
  "grief",
  "joy",
  "love",
  "nervousness",
  "optimism",
  "pride",
  "realization",
  "relief",
  "remorse",
  "sadness",
  "surprise",
  "neutral",
];

const EXAMPLES = [
  "I'm really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!",
  "It's wonderful because it's awful. At not with.",
  "Kings fan here, good luck to you guys! Will be an interesting game to watch! ",
  "I didn't know that, thank you for teaching me something today!",
  "They got bored from haunting earth for thousands of years and ultimately moved on to the afterlife.",
  "Thank you for asking questions and recognizing that there may be things that you don't know or understand about police tactics. Seriously. Thank you.",
  "100%! Congrats on your job too!",
  "Im sorry to hear that friend :(. It's for the best most likely if she didn't accept you for who you are",
  "Girlfriend weak as well, that jump was pathetic.",
  "[NAME] has towed the line of the Dark Side. He wouldn't cross it by doing something like this.",
  "Lol! But I love your last name though. XD",
  "Translation }}} I wish I could afford it.",
  "It's great that you're a recovering addict, that's cool. Have you ever tried DMT?",
  "I've also heard that intriguing but also kinda scary",
  "I never wanted to punch osap harder after seeing that However not too hardly I cant afford them taking everything away",
  "if the pain doesn't go away after 4 hours or so, it's broke.",
  "Triggered:: Welp guess it's time for me to re-up lol",
  "I'm autistic and I'd appreciate if you remove that comment. Thanks.",
];

// Reactive variables to notify the user that the model is ready.
const message = ref(0);
const initialized = ref(false);

// This is the model input.
const userText = ref(null);

// Model outputs will be stored in these reactive variables.
const topScores = ref(null);
const topLabels = ref(null);

// Holds the initialized model.
let model = null;

/* Runs the model inference.
 *
 */
const runInference = debounce(async () => {
  // Split the input text into space-separated chunks.
  const chunks = userText.value.split(" ");
  // Add some padding so to fit the model expected input.
  // This is necessary as TensorflowJS doesn't support ragged tensors.
  const padding = new Array(128 - chunks.length).fill(" ");
  // Prepare the model inputs.
  const inputs = tf.stack(RetVec.tokenizer([...chunks, ...padding]), 1);
  // Run the inference.
  const prediction = await model.executeAsync(inputs);
  // Extract all the predictions, ranked.
  const topKprediction = prediction.as1D().topk(LABELS.length);
  topLabels.value = [...topKprediction.indices.dataSync()].map(
    (i) => LABELS[i],
  );
  topScores.value = [...topKprediction.values.dataSync()];
  // Dispose of all vectors.
  inputs.dispose();
  topKprediction.values.dispose();
  topKprediction.indices.dispose();
  // Debounce this inference so that we only run it after typing has stopped.
}, 500);

// When the user types something, trigger the model.
watch(userText, async () => {
  if (!initialized.value) return;
  if (!userText.value) return;
  runInference();
});

// Load RetVec and the model at startup.
onMounted(async () => {
  message.value = "Initializing RetVec...";
  await RetVec.init(`${import.meta.env.BASE_URL}retvec_model/model.json`, 24);
  message.value = "Downloding emotions model...";
  model = await tf.loadGraphModel(
    `${import.meta.env.BASE_URL}emotion_model/model.json`,
  );
  message.value = "We're ready!";
  initialized.value = true;
  userText.value = "I enjoy hving a g00d ic3cream!!! 🍦";
});
</script>

<style scoped>
p {
  text-height: 1.2;
  margin-bottom: 0.5rem;
}

:deep(.v-chip__content) {
  display: inline-block;
  max-width: 5rem;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  text-overflow: ellipsis [...];
}
</style>
