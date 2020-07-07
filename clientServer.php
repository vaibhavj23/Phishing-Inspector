
<?php

header("Access-Control-Allow-Origin: *");
$site=	$_POST['url'];
// echo "http://" . $_SERVER['SERVER_NAME'] . $_SERVER['REQUEST_URI']; 
//echo"Hello".$site;
//echo $site;
$html = file_get_contents($site);
//echo $html;
$bytes=file_put_contents('markup.txt', $html);

// Can use this if your default interpreter is Python 2.x.
// Has some problem executing 'which python2'. So, absolute path is just simpler.
//$python_path=exec("which python 2>&1 ");
//$decision=exec("$python_path test.py $site 2>&1 ");

// Replace the path with the path of your python2.x installation.
//echo "C:\Program Files\Python37\python test.py $site 2>&1 ";

$decision=exec('C:\Users\DELL\AppData\Local\Programs\Python\Python37\python test.py '.$site.' 2>&1');
echo $decision;

?>
