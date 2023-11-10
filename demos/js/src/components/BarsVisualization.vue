<template>
  <div class="wrapper">
    <template v-for="(label, i) in labels" :key="labels[i]">
      <span>{{ labels[i] }} </span>
      <span> {{ scores[i].toFixed(2) }}</span>
      <v-progress-linear
        v-model="bars[i]"
        height="25"
        :color="i === 0 ? 'primary' : 'grey-lighten-2'"
        bg-color="grey"
        rounded-bar
        :striped="i === 0"
      >
      </v-progress-linear>
    </template>
  </div>
</template>

<script setup>
import {ref, watch, onMounted} from "vue";
const props = defineProps(["labels", "scores"]);

const bars = ref([]);

const updateBars = () => {
  bars.value = props.scores.map((s) => s * 100);
};

watch(
  () => props.scores,
  () => updateBars(),
);

onMounted(() => updateBars());
</script>

<style scoped>
.wrapper {
  display: grid;
  width: 100%;
  grid-template-columns: max-content max-content 1fr;
  grid-row-gap: 0.2rem;
  grid-column-gap: 1rem;
}
</style>
