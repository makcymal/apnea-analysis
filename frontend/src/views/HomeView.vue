<template>
  <div class="brain-map" >
    <p>sefsf</p>
<!--     
      <div class="brain-zone" v-for="zone in zones" :key="zone.name">
        <button>
          {{zone.name}}
        </button>
      </div> -->
  </div>
</template>

<script setup >
import { ref } from "vue";
const zones = ref([{
  name: 'Fp1',
  position: [0, 0]
}, {
  name: 'Fp2',
  position: [0, 0]
}, {
  name: 'F7',
  position: [0, 0]
}, {
  name: 'F3',
  position: [0, 0]
}, {
  name: 'Fz',
  position: [0, 0]
}, {
  name: 'F4',
  position: [0, 0]
}, {
  name: 'F8',
  position: [0, 0]
},{
  name: 'A2',
  position: [0, 0]
},  {
  name: 'T3',
  position: [0, 0]
}, {
  name: 'C3',
  position: [0, 0]
}, {
  name: 'Cz',
  position: [0, 0]
}, {
  name: 'C4',
  position: [0, 0]
}, {
  name: 'T4',
  position: [0, 0]
},{
  name: 'A2',
  position: [0, 0]
}, {
  name: 'T5',
  position: [0, 0]
}, {
  name: 'P3',
  position: [0, 0]
}, {
  name: 'Pz',
  position: [0, 0]
}, {
  name: 'P4',
  position: [0, 0]
}, {
  name: 'T6',
  position: [0, 0]
}, {
  name: 'O1',
  position: [0, 0]
}, {
  name: 'O2',
  position: [0, 0]
}]);




</script>
