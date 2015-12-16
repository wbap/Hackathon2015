<?php 
include '_header.php';
//$mail   = $_REQUEST['mail'];
//$sex   = $_REQUEST['sex'];
//$age   = $_REQUEST['age'];
$page_id=3;

/*print "<form name='userdata' action='register.php' method='post'>";
print "<input type='hidden' name='mail' value='".$mail."' >";
print "<input type='hidden' name='sex' value='".$sex."' >";
print "<input type='hidden' name='age' value='".$age."' >";
//numbersとnumも送ったる.
print "</form>";*/

print "<script>";
print '
$(function(){
$("#next").click(function(evt) {
              window.location.href = "review.php"
});
});

';

print "</script>";

?>


<div id="wrapper">
	<div id="featured-wrapper">
	
		<div class="extra2 container">
			<div class="ebox1">
				<div class="hexagon"><span class="icon icon-lightbulb"></span></div>
				<div class="title">
					<h2>クソコラ生成器のやり方</h2>
					<span class="byline">Instruction of Robot Fortune-Telling</span>
				</div>

				<div>
		以下の様な画面が現れるので、画像の面白さをスライド・バーでクリックし、送信ボタンを押してください<br>
		<p>
		<a href="./instruction_1.png"><img src="./instruction_1.png" width=880 height=524 ></a><br>
		</p>
		</div>
					<!--	<a id="finish" class="button">説明をスキップ</a>-->
				<a id="next" class="button">始める</a>

			</div>
							</div>		
		</div>	
	</div>
</div>


<?php
include 'log.php';
?>
</body>
</html>
