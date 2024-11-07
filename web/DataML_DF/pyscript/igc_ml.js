var modelPkl = 0;

async function selectModel() {
  
  document.getElementById("output").innerHTML = "";
  selIndex = document.IGC_ML.model.selectedIndex;
  setCookie("selectedIndex", selIndex ,1000);
  folder = (document.IGC_ML.model[selIndex].text);
  console.log(folder);
  
  fetch(folder+"/config.txt")
  .then(response => response.text())
  .then(text => {
    setCookie("features", text.trim().split(",") ,1000);
    features = getCookie("features").split(",");
    console.log("Features:  ", features);
    createEntries(features);
    setButtonLabel();
  })
  .catch(error => console.error('Error:', error));
  /*
  // Use this when opening modelPkl in JS

  fetch(folder+"/model_DF_GradientBoostingClassifier.pkl")
  .then(response => response.arrayBuffer())
  .then(buffer => {
    const byteArray = new Uint8Array(buffer);
    modelPkl = byteArray;
  })
  .catch(error => console.error('Error:', error));
  */
 }

function createEntries(features) {
  cleanup();
  fv = [];
  var br = document.createElement("break");
  br.innerHTML = "<br>";
  document.IGC_ML.appendChild(br);
  
  for (let i = 0; i < features.length; i++) {
    fv.push(i);
    var parent = document.createElement("div");
    //parent.id="entries";
    var l = document.createElement("label");
    l.textContent = features[i]+" = ";
    l.htmlFor = "Entry"+i;
    l.id = "Label"+i;

    var p = document.createElement("input");
    p.type = "text";
    p.id = "Entry"+i;
    p.setAttribute('value', fv[i]);
    p.name = features[i];

    parent.appendChild(l);
    parent.appendChild(p);
    parent.appendChild(br);
    document.IGC_ML.appendChild(parent);
    }
}

function cleanup() {
  const elements = document.querySelectorAll("div");
  elements.forEach(element => {
    if (element.id != "out" && element.id != "predict") {
        element.remove();
        }
  });
}

function setButtonLabel() {
  selIndex = document.IGC_ML.model.selectedIndex;
  folder = (document.IGC_ML.model[selIndex].text);
  const button = document.getElementById("button");
  button.innerHTML = "Predict "+folder.slice(0,5);
}

function init() {
  document.IGC_ML.model.selectedIndex = getCookie("selectedIndex");
}

//window.onload = init();

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
