<template>
  <h1>RetVec: simple demo</h1>
  <div class="status-message">{{ message }}</div>
  <input
    v-model="userInput"
    placeholder="Type a word to see its binarized version"
  />
  <div v-if="binarized" class="binarized">
    <div v-for="element in binarized" :class="{ [`el-${element}`]: true }">
      {{ element }}
    </div>
  </div>
</template>

<script setup>
import { onMounted, ref, computed } from "vue";
// We import RetVec from a local copy.
// TODO: npm install it instead, as the user would do.
import RetVec from "../../retvecjs/retvec.ts";

// We import the model here to make sure that it's available in a production build.
import modelUrl from "../../retvecjs/model/v1/model.json?url";
import modelWeightsUrl from "../../retvecjs/model/v1/group1-shard1of1.bin?url";

// Reactive elements of the page.
const message = ref(0);
const initialized = ref(false);
const userInput = ref(null);

// When the user types a word, we binarize it.
const binarized = computed(() => {
  if (!initialized.value) return;
  if (!userInput.value) return;
  return RetVec.binarizer(userInput.value).dataSync();
});

// Load RetVec at startup.
onMounted(async () => {
  message.value = "Initializing RetVec...";
  await RetVec.init(modelUrl);
  message.value = "RetVec ready!";
  initialized.value = true;
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
