/*

   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

function newWindow(pageName,wide,tall,scroll) {
  window.open(pageName,"",
    "toolbar=0,location=0,directories=0,status=0,menubar=0,scrollbars=" +
    scroll + ",resizable=0,width=" + wide + ",height=" + tall +
    ",left=0,top=0");
}

/* Document objects - An html document is classified as one of: Index,
DocBook, ModuleSpec, ModuleIndex, Latex2HTML, or ReleaseNotes.  To
inherit the correct "look and feel", an html document (whether hand
written or generated) should contain the following code in the <head>
element:

<script type="text/javascript" src="path-to/tools.js"></script>
<script type="text/javascript">var doc = new XXXXDocument();</script>

where XXXX is one of "Index", "DocBook", "ModuleSpec", ModuleIndex, or
"ReleaseNotes".  The following code should be inserted right after the
<body> tag:

<script type="text/javascript">doc.preContent();</script>

The following code should be inserted right before the </body> tag:

<script type="text/javascript">doc.postContent();</script>
*/

// Base document object
function Document() { }

Document.prototype.beginContent = function() {
  document.write("<div class=\"content\">");
}

Document.prototype.endContent = function() {
  document.write("</div>");
}

Document.prototype.doTopBanner = function() {
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

Document.prototype.doBottomBanner = function() {}

Document.prototype.preContent = function() {
  this.doTopBanner();
  this.beginContent();
}

Document.prototype.postContent = function() {
  this.endContent();
  this.doBottomBanner();
}

Document.prototype.insertLinkElement = function(cssfile) {
  document.write("<link href=\"", gSiteTop, "doc/Utilities/HTML/", cssfile, "\" type=\"text/css\" rel=\"stylesheet\"/>");
}

// Index document object
function IndexDocument() {
  Document.prototype.insertLinkElement("indexcommon.css");
}

IndexDocument.prototype = new Document();
IndexDocument.prototype.constructor = IndexDocument;

// DocBook document object
function DocBookDocument() {
  Document.prototype.insertLinkElement("srdocbook.css");
}

DocBookDocument.prototype = new Document()
DocBookDocument.prototype.constructor = DocBookDocument;

DocBookDocument.prototype.preContent = function() {
  Document.prototype.preContent();
  document.write("<div class=\"content-layer1\">\n");
}

DocBookDocument.prototype.postContent = function() {
  document.write("</div>\n");
  Document.prototype.postContent();
}

// Release notes document object
function ReleaseNotesDocument() {
  Document.prototype.insertLinkElement("releasenotes.css");
}

ReleaseNotesDocument.prototype = new Document()
ReleaseNotesDocument.prototype.constructor = ReleaseNotesDocument;

ReleaseNotesDocument.prototype.preContent = function() {
  Document.prototype.preContent();
  document.write("<div class=\"content-layer1\">\n");
}

ReleaseNotesDocument.prototype.postContent = function() {
  document.write("</div>\n");
  Document.prototype.postContent();
}

// Module spec document object
function ModuleSpecDocument() {
  Document.prototype.insertLinkElement("component.css");
}

ModuleSpecDocument.prototype = new Document()
ModuleSpecDocument.prototype.constructor = ModuleSpecDocument;

ModuleSpecDocument.prototype.preContent = function() {
  Document.prototype.preContent();
  document.write("<div class=\"content-layer1\">\n");
}

ModuleSpecDocument.prototype.postContent = function() {
  document.write("</div>\n");
  Document.prototype.postContent();
}

// Module index document object
function ModuleIndexDocument() {
  Document.prototype.insertLinkElement("moduleindex.css");
}

ModuleIndexDocument.prototype = new Document()
ModuleIndexDocument.prototype.constructor = ModuleIndexDocument;

// Latex2html generated pages
function Latex2HTMLDocument() {
}

Latex2HTMLDocument.prototype = new Document()
Latex2HTMLDocument.prototype.constructor = Latex2HTMLDocument;

Latex2HTMLDocument.prototype.preContent = function() {
  Document.prototype.preContent();
  document.write("<div class=\"content-layer1\">\n");
}

Latex2HTMLDocument.prototype.postContent = function() {
  document.write("</div>\n");
  Document.prototype.postContent();
}

// Tutorial documents?
// ...

/*
Code for generating table of contents.

The toc code is used as follows:

+ Insert the following anchor element before the content to be toc'ed:
  <a id="begin-toc">toc-title:tag-list</a> where 'toc-title' is the
  title string of the toc section and 'tag-list' is a comma separated
  (no spaces allowed) list of tags with optional class attributes
  (i.e. tag[.attr]) that are to be toc'ed.  Tags must be
  upper-case. Character ':' must be present to separate the toc-title
  from the tag list.

+ Insert the following script element after all content to be toc'ed
  (but before </body>): <script type="text/javascript">new
  Toc().build()</script>

+ Add css style rules that manifest hierarchical arrangements amongst
  entries in the toc.  Rules follow this form: p.toc-tag[-class] where
  'toc-' must be literally present, 'tag' is a tag name, and 'class'
  is an optional class attribute.

I've not tested the generation of multiple tocs for things like tables,
figures, etc. but it ought to work.
*/

function setClassAttribute(node, value) {
  node.className = value;
}

// Constructor. 
function Toc() { }

// Return a unique id number 
Toc.prototype.newIdNum = function() {
  this.idCount += 1;
  return this.idCount;
}

// Return current id number 
Toc.prototype.idNum = function() {
  if (this.idCount == 0)
    this.idCount = 1;
  return this.idCount;
}

// Return a new unique string to be used as the id of a toc
// target. 
Toc.prototype.newIdString = function() {
  var id = this.tocPrefix + String(this.newIdNum());
  return id;
}

// Return the current toc target id string in play 
Toc.prototype.idString = function() {
  return this.tocPrefix + String(this.idNum());
}

// Add, as 'node's previous sibling, an anchor node to be used as a
// toc target 
Toc.prototype.addTarget = function(node) {
   var target = document.createElement("A");
   var idString = this.newIdString();
   target.setAttribute("id", idString);
   node.parentNode.insertBefore(target, node);
//  node.setAttribute("id", this.newIdString);
}

// Add a toc entry which references its target 
Toc.prototype.addSource = function(node, cl) {
  var source = document.createElement("A");
  source.setAttribute("href", "#"+this.idString());
  var content = this.getContent(node);
  for (var i=0; i < content.length; ++i)
    source.appendChild(content[i]);
  var p = document.createElement("P");
  p.appendChild(source);
  setClassAttribute(p, cl);
  this.tocContainer.appendChild(p)
}

// Extract children of 'node', transforming
// <BR> elements to " " text elements.   Assumes <BR> elements will
// occur only as siblings of top level text nodes.
Toc.prototype.getContent = function(node) {
  var content = new Array();
  var i = 0;
  var aNode = node.firstChild;
  while (aNode != null) {
    if (aNode.nodeType == 1 && aNode.tagName == "BR")
      content[i++] = document.createTextNode(" ");
    else
      content[i++] = aNode.cloneNode(true);
    aNode = aNode.nextSibling;              
  }
  return content;
}

// Initialize the toc if necessary and then add 'node' to the toc.
// 'cl' is a string suffix that will be part of the node's class
// attribute. 
Toc.prototype.addEntry = function(node, cl) {
  this.addTarget(node);
  this.addSource(node, cl);
}

// TOC the given element.
Toc.prototype.tocThis = function(element) {
  for (var i=0; i<this.tocablesArray.length; ++i) {
    if (element.nodeName == this.tocablesArray[i].tag) {
      var classAttr;
      var classAttrNode = element.attributes.getNamedItem("class");
      if (classAttrNode == null || classAttrNode.nodeValue == "")
	classAttr = null;
      else
	classAttr = classAttrNode.nodeValue;
      if (classAttr == this.tocablesArray[i].clas) {
        className = "toc-" + element.nodeName;
        if (classAttr != null)
          className = className + "-" + classAttr;
        this.addEntry(element, className);
        return;
      }
    }
  }
}

// Return the next node in document order.
Toc.prototype.nextNode = function() {
  var cn = this.currentNode;
  var nextNode = cn.firstChild;
  if (nextNode == null)
    nextNode = cn.nextSibling;
  if (nextNode == null)
    while (true) {
      if (cn.parentNode == this.rootNode)
        return null;
      nextNode = cn.parentNode.nextSibling;
      if (nextNode == null)
        cn = cn.parentNode;
      else
        break;
    }
  this.currentNode = nextNode;
  return this.currentNode;
}

// Build a toc
Toc.prototype.build = function() {

  // Abort if <a class="begin-toc"> is missing or has empty content
  var beginToc = document.getElementById("begin-toc");
  if (beginToc == null || beginToc.firstChild == null)
    return;

  // Mark end of toc search
  document.write("<a id='endtoc'></a>")
  this.endElement = document.getElementById("endtoc");

  // Build array of toc-able elements from content of <a class="begin-toc">
  var tocablesString = beginToc.firstChild.nodeValue;
  var ta = tocablesString.split(/\:|,/g);
  // var tocTitle = ta.shift(); // Mac IE 5 doesn't support shift() (sigh).
  var tocTitle = ta[0]; ta = ta.slice(1)
  this.tocablesArray = new Array();
  for (var i=0; i<ta.length; ++i) {
    var t = ta[i].split(".");
    this.tocablesArray[i] = { tag : t[0], clas : null };
    if (t.length == 2)
      this.tocablesArray[i].clas = t[1];
  }

  // Create container for toc
  this.tocContainer = document.createElement("DIV");
  setClassAttribute(this.tocContainer, "toc");
  this.idCount = 0;
  this.tocPrefix = "toc";

  // Initialize nextNode() iterator
  this.currentNode = beginToc;
  this.rootNode = this.currentNode.parentNode;

  // Search for toc-able elements.
  // Note: Should be able to use DOM2 traversal API but it is implemented
  //  inconsistently among browsers (sigh).  Tried a recursive tree traversal
  //  algorithm but Safari puked on that (sigh).
  var node;
  while ((node = this.nextNode()) != null && (node != this.endElement)) {
    this.tocThis(node);
  }

  // Create section heading for TOC
  var startTocSearch = beginToc.nextSibling;
  var tocTitleElement = document.createElement("H1");
  setClassAttribute(tocTitleElement, "toc");
  tocTitleElement.appendChild(document.createTextNode(tocTitle));
  beginToc.parentNode.insertBefore(tocTitleElement, startTocSearch.nextSibling);
  beginToc.parentNode.insertBefore(this.tocContainer, tocTitleElement.nextSibling);
}
