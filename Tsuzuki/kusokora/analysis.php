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
  $limit = 8;
  print("<h1>table of review</h1>");
  print("<table>");
    echo '<tr><td>' .'p_id' . '</td><td>'. 'num_of_reviews' . '</td><td>' . 'ratio' .'</td></tr>';
  for($i = 1;$i<$limit+1;$i++){
	  $result1 = mysql_fetch_array(mysql_query("select count(*) from review where p_id = ".$i.";"));  	
	  $result2 = mysql_fetch_array(mysql_query("select count(*) from review where p_id = ".$i." and impression<50;"));  	
	  $ratio = ($result2[0]/$result1[0])*100;
	  echo '<tr><td>' . $i.'</td><td>'.$result1[0].'</td><td>'.$ratio . '</td></tr>';
  }

print("</table>");
?>