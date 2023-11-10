/* Demo boilerplate. * * Keeps demos to the point by hiding a little bit of
complexity. */
<template>
  <div>
    <div class="text-h2 pb-3">{{ title }}</div>
    <slot></slot>
    <v-snackbar
      :timeout="2000"
      color="primary"
      rounded="pill"
      v-model="visible"
    >
      <span class="text-white">
        {{ snackbar }}
      </span>
    </v-snackbar>

    <v-progress-linear
      :active="visible"
      indeterminate
      absolute
      bottom
      color="primary"
    ></v-progress-linear>
  </div>
</template>

<script setup>
import {ref, watch} from "vue";
const props = defineProps(["title", "snackbar"]);

const visible = ref(true);

watch(
  () => props.snackbar,
  () => {
    visible.value = !!props.snackbar;
  },
);
</script>
