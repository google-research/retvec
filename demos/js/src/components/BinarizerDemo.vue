<template>
  <Demo title="Encoding words with RetVec" :snackbar="message">
    <div class="pb-2">
      <p>
        In this demo, we showcase how RetVec transforms text into bit arrays.
        Note how making small typos or writing in l33t speak yields similar bit
        arrays.
      </p>
      <p>
        To see how we use RetVecJS to run this demo in your browser, please
        <a
          href="https://github.com/google-research/retvec/blob/main/demos/js/src/components/BinarizerDemo.vue"
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

    <div class="text-h4 pt-3 pb-5">Input</div>

    <v-text-field
      clearable
      label="Text to binarize"
      v-model="userText"
      placeholder="Type a word to see its binarized version"
      append-inner-icon="mdi-arrow-right"
    >
    </v-text-field>
    <div class="text-h4 pt-3 pb-5">Output</div>
    <div v-if="binarized" class="binarized">
      <div v-for="element in binarized" :class="{[`el-${element}`]: true}">
        {{ element }}
      </div>
    </div>
  </Demo>
</template>

<script setup>
/**
 * Binarizer  demo.
 *
 * - To see this run:
 *    https://google-research.github.io/retvec/binarizer_demo
 *
 * - To start in on your local machine or on Codespaces:
 *    https://github.com/google-research/retvec/blob/main/demos/js/README.md
 */
import {onMounted, ref, computed} from "vue";
import Demo from "@/components/Demo.vue";
// We import RetVec from a local copy.
// TODO: npm install it instead, as the user would do.
import RetVec from "../../retvecjs/retvec.ts";

// Reactive elements of the page.
const message = ref(0);
const initialized = ref(false);
const userText = ref(null);

// When the user types a word, we binarize it.
const binarized = computed(() => {
  if (!initialized.value) return;
  if (!userText.value) return;
  return RetVec.binarizer(userText.value).dataSync();
});

// Load RetVec at startup.
onMounted(async () => {
  message.value = "Initializing RetVec...";
  await RetVec.init(`${import.meta.env.BASE_URL}retvec_model/model.json`);
  message.value = "RetVec ready!";
  initialized.value = true;
  userText.value = "smiling";
});
</script>

<style scoped>
.status-message {
  color: blue;
  padding: 1rem 0;
}

input {
  width: 100%;
}

.binarized {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(1rem, 1fr));
  padding: 1rem 0;

  & > .el-0 {
    color: rgb(100, 100, 100);
  }
}
</style>
