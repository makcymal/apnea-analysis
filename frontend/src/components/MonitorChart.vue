<template>
  <Doughnut :data="data" :options="options" />
</template>


<script>



import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "vue-chartjs";
const data = {
        labels: ["VueJs", "EmberJs", "ReactJs", "AngularJs"],
        datasets: [
          {
            backgroundColor: ["#3285a8", "#51b6e0", "#27799c", "#1599d1"],
            data: [40, 20, 80, 10],
          },
        ],
      }

const options = {
        responsive: true,
        maintainAspectRatio: false,
      }
ChartJS.register(ArcElement, Tooltip, Legend);

export default {
  
  components: {
    Doughnut,
  },
  data() {
    return {
      data: {
        labels: ["VueJs", "EmberJs", "ReactJs", "AngularJs"],
        datasets: [
          {
            backgroundColor: ["#3285a8", "#51b6e0", "#27799c", "#1599d1"],
            data: [40, 20, 80, 10],
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
      },
    };
  },
};
</script>