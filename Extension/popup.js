// Purpose - This file contains all the logic relevant to the extension such as getting the URL, calling the server
// side clientServer.php which then calls the core logic.
function showIcon() {
   document.getElementById('loader').style.display ='block';
   document.getElementById('mutable').style.display = 'none';
 } 

 function hideIcon() {
   document.getElementById('loader').style.display ='none';
 }
function transfer(){	
	var tablink;
	
       
    showIcon();
	chrome.tabs.getSelected(null,function(tab) {
		//alert("Please enter your name");
	   	tablink = tab.url;
	   	//alert(tablink);
		// $("#p1").text("The URL being tested is - "+tablink);

		var xhr=new XMLHttpRequest();
		params="url="+tablink;
        // alert(params);

		var markup = "url="+tablink+"&html="+document.documentElement.innerHTML;
		xhr.open("POST","http://localhost:81/Malicious-Web-Content-Detection-Using-Machine-Learning-master/clientServer.php",false);
		xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
		xhr.send(markup);

		// while(xhr.readyState != 4){
		// 	$(".loader-wrapper").fadeIn("slow");
		// }
		 if(xhr.readyState == 4){

			hideIcon();
		}
		

		// Uncomment this line if you see some error on the extension to see the full error message for debugging.
        //alert(xhr.responseText);
		$("#result").text("The website is: "+xhr.responseText);
		//
		return xhr.responseText;
	});
}


$(document).ready(function(){
    $(".button").click(function(){	
		var val = transfer();
    });
});

chrome.tabs.getSelected(null,function(tab) {
   	var tablink = tab.url;
	$("#p1").text("The URL being tested is - "+tablink);
});
