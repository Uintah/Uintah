/*

The contents of this file are subject to the University of Utah Public
License (the "License"); you may not use this file except in compliance
with the License.

Software distributed under the License is distributed on an "AS IS"
basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
License for the specific language governing rights and limitations under
the License.

The Original Source Code is SCIRun, released March 12, 2001.

The Original Source Code was developed by the University of Utah.
Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
University of Utah. All Rights Reserved.

*/

var gSiteTop = findSiteTop();

function findSiteTop() {
  var path = location.pathname.substr(0, location.pathname.lastIndexOf("/"));
  var treeTop="";
  var base = path.substr(path.lastIndexOf("/") + 1);
  while (base != "" && base != "doc" && base != "src") {
    treeTop += "../";
    path = path.substr(0, path.lastIndexOf("/"));
    base = path.substr(path.lastIndexOf("/")+1);
  }
  if (base == "") {
    treeTop = "http://software.sci.utah.edu/";
  } else {
    treeTop += "../";
  } 
  return treeTop;
}

function newWindow(pageName,wide,tall,scroll){
  window.open(pageName,"","toolbar=0,location=0,directories=0,status=0,menubar=0,scrollbars=" + scroll + ",resizable=0,width=" + wide + ",height=" + tall + ",left=0,top=0");
}

function beginContent() {
  document.write("<div class=\"content\">");
}

function endContent() {
  document.write("</div>");
}

function doTopBanner() {
  document.write('<img class="top-banner" src="', gSiteTop, 'doc/Utilities/Figures/doc_banner04.jpg" border="0" usemap="#banner"> \
<map name="banner">\
<area href="http://www.sci.utah.edu" coords="133,103,212,124" alt="SCI Home">\
<area href="http://software.sci.utah.edu" coords="213,103,296,124" alt="Software">\
<area href="', gSiteTop, 'doc/index.html" coords="297,103,420,124" alt="Documentation">\
<area href="', gSiteTop, 'doc/Installation/index.html" coords="421,103,524,124" alt="Installation">\
<area href="', gSiteTop, 'doc/User/index.html" coords="525,103,571,124" alt="User">\
<area href="', gSiteTop, 'doc/Developer/index.html" coords="572,103,667,124" alt="Developer">\
</map>');
}

function doBottomBanner() {
}


/* Default pre and post content functions */
function preContent() {
  doTopBanner();
  beginContent();
}

function postContent() {
  endContent();
  doBottomBanner();
}

/* Pre and post content functions for DocBook documents */
function preDBContent() {
  preContent();
  document.write("<div class=\"content-layer1\">\n");
}

function postDBContent() {
  document.write("</div>\n");
  postContent();
}

/* Pre and post content functions for module spec documents */
function preMSContent() {
  preContent();
  document.write("<div class=\"content-layer1\">\n");
}

function postMSContent() {
  document.write("</div>\n");
  postContent();
}

/*
  Start of Toc object code
*/

/*
  The toc code should be used as follows:
  -Insert the following anchor element before the content to be toc'ed:
    <a id="begin-toc">tag-list</a>
   where 'tag-list' is list of tags with optional class attributes that are to be toc'ed.
   Tags themselves must be upper-case.  Class attributes may be upper or lower case.
  -Insert the following script element after all content to be toc'ed:
    <script  type="text/javascript">new Toc().build()</script>
  -Add css style rules that manifest hierarchical arrangements amongst entries in the toc.  
   Rules follow this form:  p.toc-tag-class where 'toc' must be literally present, 'tag' is a 
   tag name, and 'class' is an optional class attribute.  The '-' separators must be present.

  To do: All the toc entries ought to wrapped up in their own div.
*/

function setClassAttribute(node, value) {
  node.className = value;
}

/* Constructor. */
function Toc() { }

/* Return a unique id number */
Toc.prototype.newIdNum = function() {
  this.idCount += 1;
  return this.idCount;
}

/* Return current id number */
Toc.prototype.idNum = function() {
  if (this.idCount == 0)
    this.idCount = 1;
  return this.idCount;
}

/* Return a new unique string to be used as the id of a toc
   target. */
Toc.prototype.newIdString = function() {
  var id = this.tocPrefix + String(this.newIdNum());
  return id;
}

/* Return the current toc target id string in play */
Toc.prototype.idString = function() {
  return this.tocPrefix + String(this.idNum());
}

/* Add, as 'node's previous sibling, an anchor node to be used as a
   toc target */
Toc.prototype.addTarget = function(node) {
//   var target = document.createElement("A");
//   var idString = this.newIdString();
//   target.setAttribute("id", idString);
//   node.parentNode.insertBefore(target, node);
  node.setAttribute("id", this.newIdString);
}

/* Add a toc entry which references its target */
Toc.prototype.addSource = function(node, cl) {
  var source = document.createElement("A");
  source.setAttribute("href", "#"+this.idString());
  var text = this.getText(node);
  source.appendChild(text)
  var p = document.createElement("P");
  p.appendChild(source);
  setClassAttribute(p, cl);
  this.tocLast.parentNode.insertBefore(p, this.tocLast.nextSibling);
  this.tocLast = p;
}

/* Concat all of a node's text children into one text node while
   converting <br> elements into spaces */
Toc.prototype.getText = function(node) {
  var textNode = document.createTextNode("");
  var aNode = node.firstChild;
  while (aNode != null) {
    switch (aNode.nodeType) {
    case 1:
      if (aNode.tagName == "BR")
        textNode.appendData(" ");
      break;
    case 3:
      textNode.nodeValue += aNode.nodeValue
      break;
    }
    aNode = aNode.nextSibling;
  }
  return textNode;
}

/* Initialize the toc if necessary and then add 'node' to the toc.
   'cl' is a string suffix that will be part of the node's class
   attribute. */
Toc.prototype.addEntry = function(node, cl) {
  this.addTarget(node);
  this.addSource(node, cl);
}

/* Build a toc */
Toc.prototype.build = function() {

  /* Abort if <a class="begin-toc"> is missing or has empty content */
  this.startElement = document.getElementById("begin-toc");
  if (this.startElement == null || this.startElement.firstChild == null)
    return;
  else {
    this.idCount = 0;
    this.tocLast = this.startElement;
    this.tocPrefix = "toc";

    /* Mark end of toc */
    document.write("<a id='endtoc'></a>")
    this.endElement = document.getElementById("endtoc");

    /* Build array of toc-able elements from content of <a class="begin-toc"> */
    this.firstEntry = true;
    this.tocElement = null;
    var tocablesString = this.startElement.firstChild.nodeValue;
    var ta = tocablesString.split(/ +/);
    this.tocablesArray = new Array();
    for (var i=0; i<ta.length; ++i) {
      var t = ta[i].split(".");
      this.tocablesArray[i] = { tag : t[0], clas : null };
      if (t.length == 2)
        this.tocablesArray[i].clas = t[1];
    }

    /* Build the toc */
    var nextElement = this.startElement.nextSibling;
    while (true) {
      if (nextElement == this.endElement)
        return null;
      for (var i=0; i<this.tocablesArray.length; ++i) {
        if (nextElement.nodeType == 1) {
          var classAttr;
          var classAttrNode = nextElement.attributes.getNamedItem("class");
	  if (classAttrNode == null || classAttrNode.nodeValue == "")
	    classAttr = null;
	  else
	    classAttr = classAttrNode.nodeValue;
          if (nextElement.nodeName == this.tocablesArray[i].tag && classAttr == this.tocablesArray[i].clas) {
            var className = "toc-" + nextElement.nodeName;
	    if (classAttr != null)
	      className = className + "-" + classAttr;
	    this.addEntry(nextElement, className);
	    break;
	  }
	}
      }
      nextElement = nextElement.nextSibling;
    }
  }
}

