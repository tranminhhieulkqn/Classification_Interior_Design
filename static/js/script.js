$(document).ready(function () {
    var viewModel = {};

    viewModel.fileData = ko.observable({
        dataURL: ko.observable(),
        // base64String: ko.observable(),
    });
    viewModel.multiFileData = ko.observable({
        dataURLArray: ko.observableArray(),
    });
    viewModel.onClear = function (fileData) {
        if (confirm('Are you sure?')) {
            fileData.clear && fileData.clear();
        }
        hide_show(false);
    };
    ko.applyBindings(viewModel);

    var dataAll = null;

    $("#but_upload").click(function (e) {
        e.preventDefault();
        var formData = new FormData();
        var files = $('#file')[0].files;
        // Check file selected or not
        if (files.length > 0) {
            formData.append('file', files[0]);
            $.ajax({
                url: '/',
                type: 'POST',
                data: formData,
                contentType: false,
                processData: false,
                success: function (data) {
                    console.log('Success!');
                    dataAll = data
                    hide_show(true);
                    draw_chart(data = dataAll, model_name = $('.form-control option:selected').val())
                },
            });
        } else {
            alert("Please select a file.");
        }
    });
    $("#file").change(function () {
        hide_show(false);
    });
    $("#exampleFormControlSelect").change(function () {
        var selected = $('.form-control option:selected').val();
        draw_chart(data=dataAll, model_name=selected)
    })
    var re = 0;

    function hide_show(show) {
        x = document.getElementById("label_predicted_results");
        y = document.getElementById("form_results");
        z = document.getElementById("drag_label");
        if (show) {
            x.style.display = "flex";
            y.style.display = "block";
            z.style.display = "block";
        } else {
            x.style.display = "none";
            y.style.display = "none";
            z.style.display = "none";
        }
    }

    function draw_chart(data, model_name = "Xception") {
        $('#myChart').remove();
        $('#insert_chart').append('<canvas id="myChart" width="400" height="300"></canvas>');
        document.getElementById('predicted_results').innerHTML = 'Predicted results: ' + data[model_name + ' Predicted'];
        var canvas = document.getElementById('myChart');
        var ctx = canvas.getContext('2d');
        var myChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: data['Labels'],
                datasets: [{
                    label: 'Prediction Rate',
                    maxBarThickness: 100,
                    data: data[model_name],
                    backgroundColor: [
                        'rgba(255, 99, 132, 0.2)',
                        'rgba(54, 162, 235, 0.2)',
                        'rgba(255, 206, 86, 0.2)',
                        'rgba(75, 192, 192, 0.2)',
                        'rgba(153, 102, 255, 0.2)'
                    ],
                    borderColor: [
                        'rgba(255, 99, 132, 1)',
                        'rgba(54, 162, 235, 1)',
                        'rgba(255, 206, 86, 1)',
                        'rgba(75, 192, 192, 1)',
                        'rgba(153, 102, 255, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                scales: {
                    yAxes: [{
                        ticks: {
                            beginAtZero: true
                        }
                    }]
                }
            }
        });
    }
});