<?php 

$imp=$_POST['impression'];
//$email=$_POST['email'];
$numbers = explode(",",$_POST["playlist"]);
$now=$_POST['now'];
$num=$numbers[$now]+1;

?>


<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
</head>
<body>
<?php
//ini_set("mysql.default_socket","/var/run/mysqld/mysqld.sock ");
ini_set("date.timezone", "Asia/Tokyo");
$con = mysql_connect('localhost', 'root', 'Kusokora');

$con = mysql_connect('localhost', 'root', 'Kusokora');
if (!$con) {
  exit('データベースに接続できませんでした。');
}

$result = mysql_select_db('kusokora', $con);
if (!$result) {
  exit('データベースを選択できませんでした。');
}

$result = mysql_query('SET NAMES utf8', $con);
if (!$result) {
  exit('文字コードを指定できませんでした。');
}
//$id = mysql_query("select id FROM user where mail='${email}'");
//$data=mysql_fetch_array($id);
//$id = $data["id"];	
$d =  date("Y-m-d H:i:s", time());

$IP = $_SERVER['REMOTE_ADDR'];
$browser=$_SERVER["HTTP_USER_AGENT"];
$ipAddress=gethostbyaddr($IP);

$result = mysql_query("INSERT INTO review(p_id,impression,time,ip) VALUES('$num', '$imp','$d','$ipAddress')", $con);
//reloadして送信するとpostできるけど、redirect後にpostするとうまくいかん。。。
if (!$result) {
  exit('データを登録できませんでした。');
}
?>

</body>
</html>
