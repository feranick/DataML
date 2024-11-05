<?php

if(is_null($_POST['model'])) {
  $model_folder= 'Perf1_4-params_GBR_full_20240923';
}
else {
  $model_folder = $_POST['model'];
}
$flag = 0;

$configFile = $model_folder . '/config.txt';
$features = trim(file_get_contents($configFile, false));
$feat = explode(',',$features);
$feat_count = count(explode(',',$features));

for ($i = 0; $i < $feat_count; $i++) {
  if(is_null($_POST[strval($feat[$i])])) {
   $flag = 1;
   }
}

if($flag ==0) {
  $list = $_POST[strval($feat[0])];
  for ($i = 1; $i < $feat_count; $i++) {
    $list .= ',' . $_POST[strval($feat[$i])];
  }
}
else {
  $list = 0;
  for ($i = 1; $i < $feat_count; $i++) {
    $list .= ',' . $i;
  }
}

$flist = "'".$feat[0]."'";
for ($i = 1; $i < $feat_count; $i++) {
  $flist .= ',' . "'" .$feat[$i]."'";
}

$command = "./DataML_DF_web.py $model_folder '$list' '$features'  2>&1";
$output = shell_exec($command);
//exec($command, $output);

print "
<!DOCTYPE html>
<title> Data_ML - pho version </title>
<head>

<script>

console.log('PHP-model: " .$model_folder. "');
console.log('PHP-features: " .$features. "');
console.log('PHP-values: " .$list. "');

let lv = [$list];
let feat = [$flist];

function setCookie(name, value, days) {
  var e = new Date;
  e.setDate(e.getDate() + (days || 365));
  document.cookie = name + '=' + value + ';expires=' + e.toUTCString() + ';path=/;domain=.' + document.domain;
}

setCookie('modelName','".$model_folder."',1000);
setCookie('features',feat,1000);
setCookie('featureValues',lv,1000);

</script>

</head>

<body>
<b>DataML_DF PHP version</b><br>
";

include('igc_ml_ui.html');

print "
<text_area><pre>$output</pre></text_area>
</body>
</html>
";
?>
