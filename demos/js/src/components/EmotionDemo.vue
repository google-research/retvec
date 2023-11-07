<template>
  <div class="status-message">{{ message }}</div>
  <input
    v-model="userInput"
    placeholder="Type a word to see its binarized version"
  />
  <h2>Verdict</h2>
  <div>{{ verdict }}</div>
  <h2> Full output</h2>
  <pre>{{ fullOutput }}</pre>
</template>

<script setup>
import { onMounted, ref, watch } from "vue";
import * as tf from "@tensorflow/tfjs";
import RetVec from "../../retvecjs/retvec.ts";
import debounce from "lodash/debounce";


const RETVEC_ENCODING_SIZE = 256;
const RETVEC_STR_LENGTH = 128;

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

// Reactive elements of the page.
const message = ref(0);
const initialized = ref(false);
const userInput = ref(null);
const verdict = ref(null);
const fullOutput = ref(null);
let model = null;

const runInference = debounce(async () => {
  console.log(`running inference on ${userInput.value}`);
  let inputs = RetVec.binarizer(userInput.value, RETVEC_STR_LENGTH);

  inputs = tf.expandDims(inputs, 0); // make it a batch
  let prediction = await model.executeAsync(inputs);
  fullOutput.value = await prediction.data();
  const topKprediction = prediction.as1D().topk();
  const topIndex = topKprediction.indices.dataSync()[0];
  const topValue = topKprediction.values.dataSync()[0];
  verdict.value = `${LABELS[topIndex]} (${(topValue * 100).toFixed(2)})`;
}, 100);

watch(userInput, async () => {
  if (!initialized.value) return;
  if (!userInput.value) return;
  runInference();
});

// Load RetVec at startup.
onMounted(async () => {
  message.value = "Initializing RetVec...";
  await RetVec.init('/retvec_model/model.json', RETVEC_ENCODING_SIZE);
  message.value = "Loading model...";
  model = await tf.loadGraphModel('/emotion_model/model.json');
  message.value = "RetVec ready!";
  initialized.value = true;
  console.log(tf.getBackend());
  userInput.value = "I enjoy hving a g00d ic3cream!!! 🍦";
});
</script>
