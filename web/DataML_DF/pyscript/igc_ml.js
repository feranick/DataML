var modelPkl = 0;

function showLog() {
  const url="./";
  const dropdown = document.getElementById('model');
  const selectedText = dropdown.options[dropdown.selectedIndex].text;
  const log = document.getElementById("log");
  const importance1 = document.getElementById("importance1");
  const importance2 = document.getElementById("importance2");
  log.href = url+"/"+selectedText+"/log.txt";
  importance1.href = url+"/"+selectedText+"/model_GradientBoostingRegressor_importances_MDI.png";
  importance2.href = url+"/"+selectedText+"/model_GradientBoostingRegressor_importances_Perm.png";
  }

async function selectModel() {
  document.getElementById("output").innerHTML = "";
  selIndex = document.IGC_ML.model.selectedIndex;
  setCookie("selectedIndex", selIndex ,1000);
  folder = (document.IGC_ML.model[selIndex].text);
  console.log(folder);
  showLog();

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
  // 1. Get the dedicated container
  const container = document.getElementById('feature-entries-container');
  if (!container) {
      console.error("Fatal Error: Could not find div with id='feature-entries-container'.");
      return;
  }

  // 2. Clear only the container (replaces cleanup call)
  container.innerHTML = '';

  // 3. Create and append new entries inside the container
  fv = []; // Assuming fv is needed for default values? Example: 0..N-1
  var br = document.createElement("br"); // Create <br> once if needed between entries

  for (let i = 0; i < features.length; i++) {
      fv.push(i); // Example default value based on index
      var parent = document.createElement("div"); // Keep wrapping div if desired for styling/layout
      parent.className = "feature-entry"; // Optional: Add class for styling

      var l = document.createElement("label");
      l.textContent = features[i]+"\xA0";
      l.htmlFor = "Entry"+i;
      l.id = "Label"+i;

      var p = document.createElement("input");
      p.type = "text";
      p.id = "Entry"+i;
      p.setAttribute('value', fv[i]); // Set default value
      p.name = features[i]; // Use feature name for the input name

      parent.appendChild(l);
      parent.appendChild(p);
      // parent.appendChild(br.cloneNode()); // Add <br> after each entry if needed

      // Append the new parent div to the container
      container.appendChild(parent);
  }
  // Add a final <br> after all entries if desired
  //container.appendChild(document.createElement("br"));
}

function setButtonLabel() {
  selIndex = document.IGC_ML.model.selectedIndex;
  folder = (document.IGC_ML.model[selIndex].text);
  const button = document.getElementById("button");
  button.innerHTML = "Predict "+folder.slice(0,5);
}

function init() {
  selIndex = getCookie("selectedIndex");
  if(selIndex == -1 || document.IGC_ML.model.options.length < selIndex) {
    document.IGC_ML.model.selectedIndex = 0 }
  else {
    document.IGC_ML.model.selectedIndex = selIndex;}
  showLog();
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
