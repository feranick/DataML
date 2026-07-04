// ##########################################################
// Multi-Perf model handling
// ##########################################################
var nameSelIndexBase = transformString(window.location.pathname);
var perfGroups = {};   // { "Perf1B": [folder, folder, ...], ... }
var sortedKeys = [];   // numerically/alpha sorted Perf keys

// Rule (b): match Perf<number><optional uppercase letters>, e.g. "Perf1", "Perf1B"
function getPerfKey(folder) {
  const m = folder.match(/^Perf\d+[A-Z]*/);
  return m ? m[0] : null;
}

// Sort numerically by the number, then by the letter suffix
function perfSortCompare(a, b) {
  const ma = a.match(/^Perf(\d+)([A-Z]*)$/);
  const mb = b.match(/^Perf(\d+)([A-Z]*)$/);
  const na = parseInt(ma[1], 10), nb = parseInt(mb[1], 10);
  if (na !== nb) return na - nb;
  return ma[2].localeCompare(mb[2]);
}

function buildPerfGroups(models) {
  perfGroups = {};
  models.forEach(folder => {
    const key = getPerfKey(folder);
    if (!key) { console.warn("Skipping folder (no Perf key): " + folder); return; }
    if (!perfGroups[key]) perfGroups[key] = [];
    perfGroups[key].push(folder);
  });
  // sort folders within each group for stable dropdowns
  Object.keys(perfGroups).forEach(k => perfGroups[k].sort());
  sortedKeys = Object.keys(perfGroups).sort(perfSortCompare);
}

function createSelectors() {
  const container = document.getElementById('perf-selectors-container');
  if (!container) return;
  container.innerHTML = '';

  sortedKeys.forEach(key => {
    const div = document.createElement('div');
    div.className = 'feature-entry';

    const label = document.createElement('label');
    label.textContent = key + '\xA0';
    label.htmlFor = 'select_' + key;

    const select = document.createElement('select');
    select.id = 'select_' + key;
    select.name = key;
    select.setAttribute('data-perfkey', key);

    perfGroups[key].forEach(folder => {
      const opt = document.createElement('option');
      opt.text = folder;
      select.add(opt);
    });

    // Restore previous selection (state persistence across reloads)
    const cookieName = 'selIndex_' + key + '_' + nameSelIndexBase;
    const cookieValue = getCookie(cookieName);
    if (cookieValue !== null) {
      const parsedIndex = parseInt(cookieValue, 10);
      if (!isNaN(parsedIndex) && parsedIndex >= 0 && parsedIndex < select.options.length) {
        select.selectedIndex = parsedIndex;
      }
    }

    // Persist on change
    select.addEventListener('change', function () {
      setCookie(cookieName, select.selectedIndex, 1000);
    });

    div.appendChild(label);
    div.appendChild(select);
    container.appendChild(div);
  });
}

// Build the (shared) input parameter fields once, as the UNION (superset)
// of every model's config.txt. Each model later consumes just the subset it
// needs, matched BY NAME (see main.py). Fields are laid out in first-appearance
// order over the folders (already sorted) for a stable, deterministic layout.
async function buildFeatureEntries() {
  if (sortedKeys.length === 0) return;

  // Every model folder across all Perf groups, in stable (sorted) order
  const allFolders = [];
  sortedKeys.forEach(key => perfGroups[key].forEach(f => allFolders.push(f)));

  // Fetch all config.txt in parallel; Promise.all preserves input-array order
  const texts = await Promise.all(allFolders.map(folder =>
    fetch(folder + "/config.txt")
      .then(r => {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.text();
      })
      .catch(err => {
        console.error("config.txt load failed for " + folder + ":", err);
        return "";   // skip this folder's features; others still contribute
      })
  ));

  // Union the feature names, preserving first-appearance order
  const union = [];
  const seen = new Set();
  texts.forEach(text => {
    text.trim().split(",")
      .map(s => s.trim())
      .filter(Boolean)
      .forEach(f => {
        if (!seen.has(f)) { seen.add(f); union.push(f); }
      });
  });

  if (union.length === 0) {
    console.error("No features found in any model config.txt.");
    return;
  }

  setCookie("features", union, 1000);
  createEntries(union);
}

function createEntries(features) {
  const container = document.getElementById('feature-entries-container');
  if (!container) {
    console.error("Could not find div with id='feature-entries-container'.");
    return;
  }
  container.innerHTML = '';

  for (let i = 0; i < features.length; i++) {
    const parent = document.createElement("div");
    parent.className = "feature-entry";

    const l = document.createElement("label");
    l.textContent = features[i] + "\xA0";
    l.htmlFor = "Entry" + i;
    l.id = "Label" + i;

    const p = document.createElement("input");
    p.type = "text";
    p.id = "Entry" + i;
    p.setAttribute('value', i);   // default value = index (mirrors old behavior)
    p.name = features[i];

    parent.appendChild(l);
    parent.appendChild(p);
    container.appendChild(parent);
  }
}

// #######  Utilities  ##################################
function getCookie(name) {
  return (name = (document.cookie + ';').match(new RegExp(name + '=.*;'))) && name[0].split(/=|;/)[1];
}

function setCookie(name, value, days) {
  var e = new Date;
  e.setDate(e.getDate() + (days || 365));
  document.cookie = name + '=' + value + ';expires=' + e.toUTCString() + ';path=/;domain=.' + document.domain;
}

function transformString(input) {
  let result = input;
  if (result.endsWith('/')) result = result.slice(0, -1);
  return result.replace(/\//g, '-');
}
// ########################################################
