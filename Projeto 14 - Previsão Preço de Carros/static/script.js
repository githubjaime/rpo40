// Módulo para carregar os veículos de um fabricante

var make_select = document.getElementById('make');
var model_select = document.getElementById('model');

make_select.onchange = function () {
    make = make_select.value;

    fetch('/predict/' + make).then(function (responses) {
        responses.json().then(function (data) {
            var optionHTML = '';
            for (var model of data.models) {
                optionHTML += '<option value="' + model + '">' + model + '</option>';
            }
            model_select.innerHTML = optionHTML;
        })
    });
}