/**
 * plugins/vuetify.js
 *
 * Framework documentation: https://vuetifyjs.com`
 */

// Styles
import "@mdi/font/css/materialdesignicons.css";
import "vuetify/styles";

// Composables
import {createVuetify} from "vuetify";
import {md3} from "vuetify/blueprints";

// https://vuetifyjs.com/en/introduction/why-vuetify/#feature-guides
export default createVuetify({
  blueprint: md3,
  theme: {
    theme: {
      defaultTheme: "light",
    },
    themes: {
      light: {
        colors: {
          primary: "#620eej",
          secondary: "#03dac6",
        },
      },
    },
  },
});
