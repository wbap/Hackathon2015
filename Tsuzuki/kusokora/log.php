<?php
//ini_set("mysql.default_socket","/var/run/mysqld/mysqld.sock");
ini_set("date.timezone", "Asia/Tokyo");
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
$d =  date("Y-m-d H:i:s", time());

  $IP = $_SERVER['REMOTE_ADDR'];
  $browser=$_SERVER["HTTP_USER_AGENT"];
  $ipAddress=gethostbyaddr($IP);
  $result = mysql_query("INSERT INTO log(page_id,time,ip,browser) VALUES('".$page_id."','".$d."','".$ipAddress."','".$browser."')",$con);

if (!$result) {
  exit('データを登録できませんでした。');
}

$con = mysql_close($con);
if (!$con) {
  exit('データベースとの接続を閉じられませんでした。');
}
?>
