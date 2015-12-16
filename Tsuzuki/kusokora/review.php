<?php 
include '_header.php';
?>


<div id="wrapper">
  <div id="featured-wrapper">
  

<?php

//emailが
$imp = NULL;
  if (isset($_POST['impression'])){
    $imp = $_POST['impression'];
  }

if($imp == NULL){
  //post から取得したxに何も値が入っていなければregisterから来たのだと判断
     //registerから来た場合Get actionでパラメーターを取得 
  //$email = $_REQUEST["email"];
  $now=0;//registerから来てたら,nowは0

//  ini_set("mysql.default_socket","/var/run/mysqld/mysqld.sock ");
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
  //$id = mysql_query("select id FROM user where mail='".$email."'");

  //$data=mysql_fetch_array($id);
  //$id = $data["id"];//$idにuser_idを代入。
}
else{
  //registerからきた場合以外は、api.phpのなかのPOSTアクションでパラメーターを取得
   include 'api.php'; //ここで記録してる
   $now=$now+1;
   print '<script>';?>
   function scroll() { 
      window.scrollTo(100,500); 
    } 
    window.onload=scroll;
<?php   print '</script>';
   if ($now==(count($numbers))){//8個目の動画ならredirect
    $url='./result.php';
    print "<script language='javascript'>";
    print "location.href='".$url."'";
    print "</script>";
   }
}

//echo $id;
if($imp == NULL){
  $mode = 0;
  //mode: 0 動画数が少ない時。　mode: 1 動画数が多い時
  if($mode==0){
    $coin = mt_rand(0,1);
    if($coin == 0){
      $numbers = array(0,2,4,6);
    }
    else{
      $numbers = array(1,3,5,7);
    }
  }
  else if($mode==1){
    $n = 29;
    //画像の数
    $s = 5;
    //一回のテストで見る画像の数
    $c = floor($n/$s);
    $num_of_reviews = array();
    for($k = 0; $k < $c ;$k++ ){
      $num_p = mysql_query("select count(*) FROM review where p_id='".$k."'");   
      $data=mysql_fetch_array($num_p);
      //print_r($data);
        array_push($num_of_reviews, $data[0]);
    }
      //print_r($num_of_reviews);
    $flag = 0;
    for($k=0;$k<10;$k++){
        for($i = 0;$i < $c;$i++){
          $reviewed_flag = mysql_fetch_array(mysql_query("select count(*) FROM review where p_id='".($i+$c)));   
            if($num_of_reviews[$i]<(5*$k+5)){
           //  print("reviewed_flag ".$reviewed_flag[0]."<br>");
            if($reviewed_flag[0]==0){
              //今まで評価されていない
              $flag = 1;
              $base = $i;
              $numbers = array();
              for ($j=0;$j<$s;$j++){
                if($c*$j+$base<$n){
                  array_push($numbers,$c*$j+$base) ;
                }
              }
              break;
            }
            else{
            //reviewed_flagの値が最小の群を選択

            }
          }
      }
    }
    if($flag==0){
      $url='./result.php?id='.$id;
      print "<script language='javascript'>";
      print "location.href='".$url."'";
      print "</script>";
    }
  }
}

//print_r($numbers);
$con = mysql_close($con);
if (!$con) {
  exit('データベースとの接続を閉じられませんでした。');
}

?>
<?php
//print_r($numbers);
$post_numbers=implode(",",$numbers);
print 'Count: '.($now+1).'/'.count($numbers).'<br>';
$page_id=4+$now;
print "クソコラを見て、面白さを評価してください。<br>※極稀に動画の読み込みに時間がかかることがあります。読み込まない場合はリロードしてください。<br>";
print "<img src='./pictures/image" ;
print ($numbers[$now]+1);
print ".jpg' width='400' height='300' ></video>";

?>
<br>
面白さを0-100で評価してください。(100に近づくほどおもしろいことを意味します。)<br>
<label><input id="range" type="number" name="range" min="0" max = "100"></label>
<input id="submit_button" type="submit" name="submit_button" value="送信">

<?php
print "<form name='kusokoradata' action='review.php' method='post'>";
//print "<input type='hidden' name='email' value='".$email."' >";
print "<input type='hidden' name='impression' value='' >";
print "<input type='hidden' name='now' value=".($now)." >";
print "<input type='hidden' name='playlist' value='".$post_numbers."' >";
//numbersとnumも送ったる.
print "</form>";

?>
<script>
$(function(){
  $("#submit_button").click(function(evt) {
           var rng = document.getElementById("range");
           kusokoradata.impression.value = 100 - rng.value;
           kusokoradata.submit();
});


});


</script>

  </div>
</div>


<?php
include 'log.php';
?>
</body>
</html>











