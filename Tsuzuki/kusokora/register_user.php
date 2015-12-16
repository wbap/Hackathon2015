<?php 
include '_header.php';
$page_id=2;

?>
<div id="wrapper">
	<div id="featured-wrapper">
	
		<div class="extra2 container">
			<div class="ebox1">
				<div class="hexagon"><span class="icon icon-lightbulb"></span></div>
				<div class="title">
					<h2>クソコラ生成器を始める</h2>
					<span class="byline">Instruction of Kusokora Generator</span>
				</div>
				<center>
					<p>
<div style="border-style: dotted;border-width:3px;width:600px;border-radius: 10px;box-shadow: 5px 5px 5px #AAA;">
以下のフォームを埋めて頂き、送信を押すとクソコラ生成がスタートします。<br>
クソコラ5個を評価してもらいます。<br>
(入力していただいたメールアドレスは研究目的以外では使用しません。)<br>
</div>
</p>
</center>
<br>
<div>
<form action="instruction1.php" method="post">
  メールアドレス<br>
  <input type="text" style="width:300px;font-size:20px;" name="mail"><br><br>
性別<br>
<div style="font-size:30px;"><input type="radio" name="sex" style="width:30px;height:30px;" value="0"> 
男
<input type="radio" name="sex" style="width:30px;height:30px;" value="1"> 
女</div>
<br><br>
  年齢<br>
  <input type="text" style="width:100px;font-size:20px;" name="age"><br><br>

  <input class="button" type="submit">
  
</form>
</div>
		</div>	
	</div>
</div>
</div>
<div id="copyright" class="container">
	<p>Copyright (c) 2015 Kusano Hitoshi All rights reserved. | Templates by <a href="http://fotogrph.com/">Fotogrph</a> | Design by <a href="http://www.freecsstemplates.org/" rel="nofollow">FreeCSSTemplates.org</a>.</p>
</div>
<?php
include 'log.php';
?>
</body>
</html>
