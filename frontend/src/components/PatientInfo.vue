<template>
    <div class="container back-container" style="padding: 0%;">
        <div class="container back-container-info">
            <!-- тут надо данные подсасывать с бэкенда -->
            <h3>Данные о пациенте</h3>
            ФИО/АЙДИ: <br>
            Пол: <br>
            Рост: <br>
            Вес: <br>
            Пульс: <br>
            Другие данные: <br>
        </div>
        <div class="container back-container-info">
            <h3>Результат анализа ЭЭГ на АПНОЭ</h3>
            <!-- тут надо данные подсасывать с бэкенда -->
            Вероятность АПНОЭ: 74%
        </div>
        

    </div>
  </template>
  
  <script setup >
  
  </script>
  
  
<style lang="scss">
 .back-container {
    background-color: #e0fbfc;
    min-height: 80vh;
    border: 2px solid #cecece;
 }

 .back-container-info {
    background-color: #e0fbfc;
    min-height: 40vh;
    border: 2px solid #cecece;
 }


 .back-table {
    background-color: beige;
 }
</style>
  