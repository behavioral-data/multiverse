   function disp_prompt()
   {
    return;
   }

function delete_alt(x){
  var div = x.parentElement;
  div.style.display = "none";
}



  function show_alt(x){
    x.getElementsByClassName("dropdown-content")[0].style.display='block';
    console.log('hello');
  }


  function newElement(x){
    inputValue = x.parentNode.getElementsByTagName("input")[0].value;
    var t = document.createTextNode(inputValue);
    var li = document.createElement("li");
    li.appendChild(t);
    if (inputValue === '') {
    alert("You must write something!");
    return;
    }
    x.parentNode.getElementsByTagName("input")[0].value = "";
    var span = document.createElement("span");
    var txt = document.createTextNode("\u00D7");
    span.className = "close";
    span.setAttribute("onclick", "delete_alt(this)");
    span.appendChild(txt);
    li.appendChild(span);
    x.parentNode.appendChild(li);
    // debugger;
  }

      function gText(e) {
        let range = document.getSelection().getRangeAt(0);
        if (range.toString().length==0){
          console.log('hello');
          return;
        }
        // debugger;
        // start_node = (range.startContainer.nextSibling  && range.startContainer.nextSibling.nodeType!=1)? range.startContainer: range.startContainer.parentNode;
        start_node = (range.startContainer.parentElement.nodeName=="PRE")? range.startContainer:range.startContainer.parentNode;
        end_node = (range.endContainer.parentElement.nodeName=="PRE")? range.endContainer:range.endContainer.parentNode;
        // end_node = (range.endContainer.previousSibling  && range.endContainer.nextSibling.nodeType!=1)? range.endContainer: range.endContainer.parentNode;
        let node = start_node;
        let parent_node = node.parentNode;
        let flag = false;
        while(true){
          // debugger;
          if (node.isSameNode(end_node)){
            flag = true;
          }
          // only change class of "span"
          if (node.nodeName!="#text"){
            node.setAttribute("class", (node.getAttribute("class")==node.getAttribute("init_class"))? "dp":node.getAttribute("init_class"));
            if (node.getAttribute("class")=="dp" && node.children.length==0){
              // add alternatives (dropdown-content)
              new_node = document.createElement("ul");
              new_node.setAttribute("class", "dropdown-content");
              input = document.createElement("input");
              input.setAttribute("type", "text");
              input.setAttribute("id", "myInput");
              // new_node.appendChild(input);

              add = document.createElement("span");
              add.setAttribute("onclick","newElement(this)");
              add.setAttribute("class", "addBtn");
              add.textContent = "Add";
              // new_node.appendChild(add);

              for (i=0; i<3; i++){
                son = document.createElement("li");
                son.textContent = "Link"+i;
                grand_son = document.createElement("span");
                grand_son.textContent = "x";
                grand_son.setAttribute("class", "close");
                grand_son.setAttribute("onclick", "delete_alt(this)");
                // son.appendChild(grand_son);
                // new_node.appendChild(son);
              }
              // node.appendChild(new_node);
            }
          }
          if (flag){break;}
          node= (node.nextSibling)?node.nextSibling:node.parentNode;
        }
      }
      document.onmouseup = gText;
      if (!document.all) document.captureEvents(Event.MOUSEUP);
