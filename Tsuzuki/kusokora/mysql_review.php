<?php 
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
  $result = mysql_query("select * FROM review");
  print("<h1>table of review</h1>");
  print("<table>");
    echo '<tr><td>' .'id' . '</td><td>'. 'p_id' . '</td><td>' . 'impression' .'</td><td>' . 'date'.'</td><td>' . 'ip' .'</td></tr>';
  while ($data = mysql_fetch_array($result)) {
    echo '<tr><td>' . $data['id'] . '</td><td>' .$data['p_id'] . '</td><td>' . $data['impression'].'</td><td>'. $data['time'].'</td><td>'. $data['ip'] . '</td></tr>';
  }
  print("</table>");

?>
<style>
table {
  border-collapse: collapse;
}
td {
  border: solid 1px;
  padding: 0.5em;
}
</style>