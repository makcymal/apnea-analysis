<template>
<header>
  <p>Пациент</p>
  <p>Период</p>
</header>
<body>
  <div class="container">
    <div class="btn-group">
  <button type="button" class="btn btn-danger dropdown-toggle" data-bs-toggle="dropdown" aria-expanded="false">
    Действие
  </button>
  <ul class="dropdown-menu">
    <li><a class="dropdown-item" href="#">Действие</a></li>
    <li><a class="dropdown-item" href="#">Другое действие</a></li>
    <li><a class="dropdown-item" href="#">Что-то еще здесь</a></li>
    <li><hr class="dropdown-divider"></li>
    <li><a class="dropdown-item" href="#">Отделенная ссылка</a></li>
  </ul>
</div>
<div class="container">
    <div class="row align-items-start">
    <div class="col">
      <brain-map />
    </div>
    <div class="col">
      <patient-info />
    </div>
    </div>
  </div>
</div>
</body>
</template>

<script setup >
import { ref , createApp} from "vue";
import BrainMap from './components/BrainMap'
import PatientInfo from './components/PatientInfo'
const app = createApp({})

// Регистрация компонента
app.component('brain-map', BrainMap)

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
header {
  display: flex;
}
.wrapper {

  display: flex;
  align-items: center;
  flex-direction: column;
    // height: 500px;
  width: 500px;
}
.brain-map {


  display: flex;
  align-items: center;
  justify-content: center;
    border: 1px #006eff solid;
  border-radius: 50%;
  height: 600px;
  width: 600px;
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
  transform: translateY(-3px)
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
