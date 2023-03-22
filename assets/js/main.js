console.log("Hi bitch im here!");

var sidebarContainer = document.getElementById("sidebar");
var btns = sidebarContainer.getElementsByClassName("sidebar-item");
var current = document.getElementsByClassName("active");
current[0].className = current[0].className.replace(" active", "");
var title = document.getElementsByTagName("title")[0].innerText;
if (title == ' Home '){
    btns[0].className += " active";
}
else if (title == ' Blog '){
    btns[1].className += " active";
}
else if (title == ' Projects '){
    btns[2].className += " active";
}
else if (title == ' Books library '){
    btns[3].className += " active";
}
else if (title == ' About '){
    btns[4].className += " active";
}
else {
    console.log(`Reading ${title}`)
}
