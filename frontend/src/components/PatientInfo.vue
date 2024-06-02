<template>
        <div class="container back-container-info">
            <!-- тут надо данные подсасывать с бэкенда -->
            <h3>Данные о пациенте</h3>
            ФИО/АЙДИ: <br>
            Пол: М<br>
            Рост: 182<br>
            Вес: 108<br>
            Возраст: 35 <br>
            Пульс: 60<br>
            BPsys: 140<br>
            BPdia: 90<br>
            ODI: 34,6<br>
        </div>
        <div class="container back-container-info">
            <h3>Результат анализа ЭЭГ</h3>
            <!-- тут надо данные подсасывать с бэкенда -->
            Вероятность апноэ: 88%
    </div>
    <div class="container">
    <div class="row align-items-start">
    <div class="col">
      <div class="div" >
        <img src="@/images/g1.jpg" width="150%" height="150%" align="left" hspace="10%" vspace="10%" />
      </div>

    </div>
    <div class="col">
      <img src="@/images/g2.jpg" width="80%" align="right" hspace="10%" vspace="10%" />
    </div>
    </div>
  </div>
  </template>
  
  <script setup >
  
  </script>
  
  
<style lang="scss">
 .back-container {
    background-color: #e0fbfc;
    min-height: 60vh;
    border: 2px solid #cecece;
 }

 .back-container-info {
    background-color: #e0fbfc;
    min-height: 12vh;
    vertical-align: bottom;
    border: 2px solid #cecece;
 }


 .back-table {
    background-color: beige;
 }
</style>
  