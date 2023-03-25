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

(function() {
    "use strict";
      const select = (el, all = false) => {
          el = el.trim()
          if (all) {
              return [...document.querySelectorAll(el)]
          } else {
              return document.querySelector(el)
          }
      }
      const typed = select('.typed')
      console.log(typed);
      if (typed) {
      let typed_strings = typed.getAttribute('data-typed-items')
      typed_strings = typed_strings.split(',')
      new Typed('.typed', {
          strings: typed_strings,
          loop: true,
          typeSpeed: 100,
          backSpeed: 50,
          backDelay: 2000
      });
  }})();
