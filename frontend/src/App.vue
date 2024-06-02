<template>
<div>
<header>
<select name="select">

  <option value="value1" selected>Иванов Иван Иванович</option>
  <option value="value2" >Годунов Иван Иванович</option>
  <option value="value3">Петров Иван Иванович</option>
</select>
<select name="select">

  <option value="value1" selected>Эксперимент 1</option>
  <option value="value2" >Эксперимент 2</option>
  <option value="value3">Эксперимент 3</option>
</select>


</header>
<body>
  <div class="container">
    <div class="btn-group">

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
      <brain-map  />
    </div>
    <div class="col">
      <patient-info />
    </div>
    </div>
  </div>
</div>
</body>
<div class="btn" >
      <button class="create" >Сгенерировать</button>

</div>
<!-- <div class="graphs" >
<div class="graph">
<monitor-chart> </monitor-chart>

</div>
<div class="graph">
<scatter-chart> </scatter-chart>

</div>
</div> -->

</div>

</template>

<script setup >
import { ref , createApp} from "vue";
import BrainMap from './components/BrainMap'
import PatientInfo from './components/PatientInfo'
import MonitorChart from './components/MonitorChart'
import ScatterChart from './components/ScatterChart'

const app = createApp({})

// Регистрация компонента
app.component('brain-map', BrainMap, MonitorChart, ScatterChart)

</script>


<style lang="scss">
.btn {
  width: 100%;
  display: flex;
  justify-content: center;
}
.create {

}
.graphs {
  display: flex;
  justify-content: space-around;
}
.graph {
  max-width: 600px;
}
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
  justify-content: space-around;
  padding: 20px 0;
}
select {
  border-radius: 10px;
}

.wrapper {

  display: flex;
  align-items: center;
  flex-direction: column;
    // height: 500px;
  width: 500px;
}
.create {
  transition: 0.3s;
}
.create:hover {
  transform: translateY(-4px);

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
// .brain-zone {
//   cursor: pointer;
//   display: flex;
//   align-items: center;
// justify-content: center;
//   width: 35px;
//   height: 35px;
//   margin: 25% 10%;
//     border: 1px #006eff solid;
//   border-radius: 50%;
//   transition: 0.3s;
// }
.brain-zone:hover {
  transform: translateY(-3px)
}

.button-zone {
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
