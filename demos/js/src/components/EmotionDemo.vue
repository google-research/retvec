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

/*
array([b'I\xe2\x80\x99m really sorry about your situation :( Although I love the names Sapphira, Cirilla, and Scarlett!',
       b"It's wonderful because it's awful. At not with.",
       b'Kings fan here, good luck to you guys! Will be an interesting game to watch! ',
       b"I didn't know that, thank you for teaching me something today!",
       b'They got bored from haunting earth for thousands of years and ultimately moved on to the afterlife.',
       b'Thank you for asking questions and recognizing that there may be things that you don\xe2\x80\x99t know or understand about police tactics. Seriously. Thank you.',
       b'You\xe2\x80\x99re welcome', b'100%! Congrats on your job too!',
       b'I\xe2\x80\x99m sorry to hear that friend :(. It\xe2\x80\x99s for the best most likely if she didn\xe2\x80\x99t accept you for who you are',
       b'Girlfriend weak as well, that jump was pathetic.',
       b"[NAME] has towed the line of the Dark Side. He wouldn't cross it by doing something like this.",
       b'Lol! But I love your last name though. XD',
       b'Translation }}} I wish I could afford it.',
       b"It's great that you're a recovering addict, that's cool. Have you ever tried DMT?",
       b"I've also heard that intriguing but also kinda scary",
       b'I never wanted to punch osap harder after seeing that However not too hardly I cant afford them taking everything away',
       b'The thought of shooting anything at asylum seekers is appalling.',
       b"if the pain doesn't go away after 4 hours or so, it's broke.",
       b"Triggered:: Welp guess it's time for me to re-up lol",
       b"I'm autistic and I'd appreciate if you remove that comment. Thanks."],
      dtype=object)>
      */

// Reactive elements of the page.
const message = ref(0);
const initialized = ref(false);
const userInput = ref(null);
const verdict = ref(null);
const fullOutput = ref(null);
let model = null;

const runInference = debounce(async () => {
  console.log(`running inference on ${userInput.value}`);
  const splits = userInput.value.split(' ');
  let inputs = RetVec.tokenizer(userInput.value.split(' ') , 16);
  console.log('in', inputs, inputs[0].shape)
  inputs = tf.expandDims(inputs, 0); // make it a batch
  console.log('in', inputs.shape)
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
  await RetVec.init(`${import.meta.env.BASE_URL}retvec_model/model.json`, 24);
  message.value = "Loading model...";
  model = await tf.loadGraphModel(`${import.meta.env.BASE_URL}emotion_model/model.json`);
  message.value = "RetVec ready!";
  initialized.value = true;
  console.log(tf.getBackend());
  userInput.value = "I enjoy hving a g00d ic3cream!!! 🍦";
});
</script>
