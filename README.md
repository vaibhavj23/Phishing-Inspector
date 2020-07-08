# Phishing_Inspector
#### NOTE - 
#### You must have python installed in your PC along with all the required packages needed for this project.

### Steps you will need to perfrom to run the project -
* Move the project folder to the localhost location. For eg. ```D:/wamp64/www/project``` in case of Wamp Server.
* Modify the path of your Python 3.x installation in ```clientServer.php```.
* Then modify the path for clientServer.php in popup.js file present in Extension folder. For eg. ```xhr.open("POST","http://localhost:81/Project_Name/clientServer.php",false);```.
* Then go to ```chrome://extensions```, click on load unpacked and select the 'Extension' folder from the project.
* Once you are done performing the above steps , you can check any website by clicking the check button in the extension present at the top right panel of chrome window. 
