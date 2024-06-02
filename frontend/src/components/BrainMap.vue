<template>
  <div class="brain-map">
    <div class="wrapper">
      <div class="brain-line" v-for="line in zones">
        <div
          class="brain-zone"
          v-for="zone in line"
          :key="zone.name"
          @click="changeColor(zone.name)"
           :style="[stateZones[zone.name] ? {'background': '#0099ff'} : {'background': '#FFF'}]"
        >
          <button >
            {{ zone.name }}
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup >
import { ref,reactive } from "vue";
const flag = ref(0);
const stateZones = reactive({})
const zones = ref([
  [
    {
      name: "Fp1",
      position: [0, 0],
    },
    {
      name: "Fp2",
      position: [0, 0],
    },
  ],
  [
    {
      name: "F7",
      position: [0, 0],
    },
    {
      name: "F3",
      position: [0, 0],
    },
    {
      name: "Fz",
      position: [0, 0],
    },
    {
      name: "F4",
      position: [0, 0],
    },
    {
      name: "F8",
      position: [0, 0],
    },
  ],
  [
    {
      name: "A1",
      position: [0, 0],
    },
    {
      name: "T3",
      position: [0, 0],
    },
    {
      name: "C3",
      position: [0, 0],
    },
    {
      name: "Cz",
      position: [0, 0],
    },
    {
      name: "C4",
      position: [0, 0],
    },
    {
      name: "T4",
      position: [0, 0],
    },
    {
      name: "A2",
      position: [0, 0],
    },
  ],
  [
    {
      name: "T5",
      position: [0, 0],
    },
    {
      name: "P3",
      position: [0, 0],
    },
    {
      name: "Pz",
      position: [0, 0],
    },
    {
      name: "P4",
      position: [0, 0],
    },
    {
      name: "T6",
      position: [0, 0],
    },
  ],
  [
    {
      name: "O1",
      position: [0, 0],
    },
    {
      name: "O2",
      position: [0, 0],
    },
  ],
]);

function changeColor(name) {
  this.stateZones[name] = !this.stateZones[name];
  // emit('changeStateZone', )
}

function isActive(params) {

  return this.stateZones[name] || false;
}
const props = defineProps({
  stateZones: Object,
});
</script>


<style lang="scss">
html,
body {
  height: 100%;
}

* {
  margin: 0;
  box-sizing: border-box;
}
.wrapper {
  display: flex;
  align-items: center;
  flex-direction: column;
  // height: 500px;
  // width: 200px;
}
.brain-map {
  display: flex;
  align-items: center;
  justify-content: center;
  border: 1px #006eff solid;
  border-radius: 50%;
  height: 300px;
  width: 300px;
}
.brain-zone {
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  width: 35px;
  height: 35px;
  margin: 25% 10%;
  border: 1px #006eff solid;
  border-radius: 50%;
  transition: 0.3s;
}
.brain-zone:hover {
  transform: translateY(-3px);
}

button {
  width: 35px;
  height: 35px;
  cursor: pointer;
  background: none;
  border: none;
}
.brain-line {
  display: flex;
  justify-content: center;
}
</style>
