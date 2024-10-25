function selectModel() {
  setCookie("selectedIndex", document.IGC_ML.model.selectedIndex ,1000);
  document.IGC_ML.submit();
 }

function createEntries() {
  fv = getCookie("featureValues").split(",");
  features = getCookie("features").split(",");
  for (let i = 0; i < features.length; i++) {
    var parent = document.createElement("div");
    var l = document.createElement("label");
    l.textContent = features[i]+" = ";
    l.htmlFor = "Entry"+i;
    l.id = "Label"+i;

    var p = document.createElement("input");
    p.type = "text";
    p.id = "Entry"+i;
    p.setAttribute('value', fv[i]);
    p.name = features[i];

    var br = document.createElement("br");
    br.innerHTML = "<br>";

    parent.appendChild(l);
    parent.appendChild(p);
    parent.appendChild(br);
    document.IGC_ML.appendChild(parent);
    }

  var parent = document.createElement("div");
  var b = document.createElement("input");

  b.type = "submit";
  b.id = "Predict";
  b.value = "Predict "+getCookie("modelName").slice(0,5);
  b.name = "predict";
  //b.setAttribute("onclick", clickButton());

  parent.appendChild(br);
  parent.appendChild(b);
  document.IGC_ML.appendChild(parent);
}

function init() {
  document.IGC_ML.model.selectedIndex = getCookie("selectedIndex");
}

window.onload = init;

// #######  Utilities  ##################################
function getCookie(name) {
  return (name = (document.cookie + ';').match(new RegExp(name + '=.*;'))) && name[0].split(/=|;/)[1];
}

function setCookie(name, value, days) {
  var e = new Date;
  e.setDate(e.getDate() + (days || 365));
  document.cookie = name + '=' + value + ';expires=' + e.toUTCString() + ';path=/;domain=.' + document.domain;
}
// ########################################################
