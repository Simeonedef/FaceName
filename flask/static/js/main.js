$(document).ready(function () {

$("#input-id").fileinput({showCaption: false, dropZoneEnabled: false,
                          maxFileSize: 1000,
                          theme: "fas",
                          uploadLabel: "Guess my name",
                          uploadIcon: "<i class=\"fas fa-check-circle\"></i>",
                          uploadClass: "btn btn-success",
                          browseLabel: "Select a picture"
                            });
document.getElementById("input-id").onclick = function() {clear_results()};

})

function clear_results() {
    document.getElementById("results").innerHTML = "";
}