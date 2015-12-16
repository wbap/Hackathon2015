<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=utf-8" />
<title>登録</title>
</head>
<body>
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
$mail   = $_REQUEST['mail'];
$sex   = $_REQUEST['sex'];
$age   = $_REQUEST['age'];
if(!filter_var($mail, FILTER_VALIDATE_EMAIL)){
     exit('メールアドレスがおかしいです。');
}

$result = mysql_query("select id FROM user where mail='".$mail."'");
$id=NULL;
while($row = mysql_fetch_array($result,MYSQL_NUM)){
	$id=$row[0];
}
mysql_free_result($result);

if ($id!=NULL){
print "既に登録されたアドレスです。<br>ページが移るまで少しお待ちください";
$url='./review.php?email='.$mail;
print "<script language='javascript'>";
print "location.href='".$url."'";
print "</script>";
}
if($id==NULL){
$result = mysql_query("INSERT INTO user(mail, sex, age) VALUES('$mail', '$sex', '$age')", $con);
}
if (!$result) {
  exit('データを登録できませんでした。');
}

$con = mysql_close($con);
if (!$con) {
  exit('データベースとの接続を閉じられませんでした。');
}
print "登録しました！！<br>ページが移るまで少しお待ちください";

$url='./review.php?email='.$mail;
print "<script language='javascript'>";
print "location.href='".$url."'";
print "</script>";
?>

</p>


</body>
</html>
