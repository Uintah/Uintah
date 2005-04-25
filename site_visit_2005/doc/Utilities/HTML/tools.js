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
  document.write("<div id=\"content\">");
}

Document.prototype.endContent = function() {
  document.write("</div>");
}

Document.prototype.doTopBanner = function() {
  document.write('<img class="top-banner" src="', gSiteTop, 'doc/Utilities/Figures/doc_banner04.jpg" usemap="#banner"/>\
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
  document.write("<div id=\"content-layer1\">\n");
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
  document.write("<div id=\"content-layer1\">\n");
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
  document.write("<div id=\"content-layer1\">\n");
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
  document.write("<div id=\"content-layer1\">\n");
}

Latex2HTMLDocument.prototype.postContent = function() {
  document.write("</div>\n");
  Document.prototype.postContent();
}

/*
  A few utilites used mainly to hide IEs misbehaviors.
*/

// Set a node's class attribute value.
function setClassAttribute(node, value) {
  node.className = value;
}

// Return a node's class attribute value.
function getClassAttribute(node) {
  return getAttrValue(node, "class");
}

// A wrapper around node.attributes.getNamedItem - The standard
// specifies returning a null result if an attribute does not exist
// and standard conforming browsers (Mozilla, Safari) do so.  But not
// our friend I.E.  It returns "".
function getAttrNode(node, attrName) {
  var result = null;
  var attrNode = node.attributes.getNamedItem(attrName);
  if (attrNode == null || attrNode.nodeValue == "")
    result = null;
  else
    result = attrNode;
  return result;
}

// Return the value of a node's attribute or null if the attribute
// does not exist.
function getAttrValue(node, attrName) {
  var attrNode = getAttrNode(node, attrName);
  return attrNode == null ? null : attrNode.nodeValue;
}


/*
  Traversal code
*/

/*

Visitor is an object for visiting nodes in document order.  Visitor's
constructor takes three forms:

Visitor() -- Object is initialized to visit all nodes within the document's 
"content" div.

Visitor(start) -- Object is initialized to visit all nodes subsequent to
"start" but still within the document's "content" div.  "start" is assumed
to be within the document's "content" div.

Visitor(start, end) -- Object is initialized to visit all nodes
subequent to "start" and up to but not including "end".  Start and end
are assumed to be within the document's "content" div.

*/

// Constructor.
function Visitor(start, end) {
  this.current = null;
  if (arguments.length == 0) {
    this.current = document.getElementById("content");
    this.end = this.current.parentNode;
    this.mode = 0;
  } else if (arguments.length == 1) {
    this.current = start;
    this.end = document.getElementById("content").parentNode;
    this.mode = 0;
  } else if (arguments.length == 2) {
    this.current = start;
    this.end = end;
    this.mode = 1;
  }
}

// Return next node in document order.
Visitor.prototype.next = function() {
  var cn = this.current;
  var next = cn.firstChild;
  if (next == null)
    next = cn.nextSibling;
  if (next == null)
    while (true) {
      if (this.mode == 0 && cn.parentNode == this.end)
	return null;
      next = cn.parentNode.nextSibling;
      if (next == null)
	cn = cn.parentNode;
      else
	break;
    }
  if (this.mode == 0)
    this.current = next;
  else
    this.current = (next == this.end ? null : next);
  return this.current;
}

/*
  Code for constructing context-based tables of contents.  See 
  http://www.cvrti.utah.edu/js/doc/toc-doc.html for documentation.
*/

/*
  Code for constructing context-based tables of contents.  See file
  toc-doc.html for use instructions.
*/

// Constructor.
function Toc() { }

// Return a unique id number.
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

// Return a new unique string to be used as the id of a toc target.
Toc.prototype.newIdString = function() {
  var id = this.tocPrefix + String(this.newIdNum());
  return id;
}

// Return the current toc target id string in play
Toc.prototype.idString = function() {
  return this.tocPrefix + String(this.idNum());
}

// Add, as 'element's previous sibling, an anchor element to be used as a
// toc target
Toc.prototype.addTarget = function(node) {
   var target = document.createElement("A");
   var idString = this.newIdString();
   target.setAttribute("id", idString);
   node.parentNode.insertBefore(target, node);
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
  this.tocBody.appendChild(p)
}

// Extract children of 'node', transforming <BR> elements to " " text
// elements.  Assumes <BR> elements will occur only as siblings of top
// level text nodes.
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

// Add node to toc.  
Toc.prototype.addEntry = function(node, cl) {
  this.addTarget(node);
  this.addSource(node, cl);
}

// Parse the given node spec string.  This is a hack as it
// assumes the node spec string is valid.
Toc.prototype.parseNodeSpec = function(nodeSpecString) {
  var nodeSpec = new Object();
  nodeSpec.tag = "";
  nodeSpec.decor = null;
  nodeSpec.decorValue = null;
  var str = "";
  part = "tag";
  for (var i=0; i<nodeSpecString.length; ++i) {
    var c = nodeSpecString.charAt(i);
    if (c == '.' || c == '#') {
      nodeSpec.decor = c;
      part = "decorValue";
      nodeSpec[part] = "";
    } else {
      nodeSpec[part] += c;
    }
  }
  return nodeSpec;
}

// Parse the given rule string.  Returns an array of alternating
// (unparsed) node spec strings and op strings.  This is a hack as it
// assumes the rule string is valid.
Toc.prototype.parseRule = function(ruleString) {
  var ruleArray = new Array();
  var str = "";
  for (var i=0; i<ruleString.length; ++i) {
    var c = ruleString.charAt(i);
    if (c == ' ' || c == '>' || c == '+') {
      ruleArray.push(str);
      ruleArray.push(c);
      str = "";
    } else {
      str += c;
    }
  }
  ruleArray.push(str);
  return ruleArray;
}

// Return a compiled rule.  A compiled rule is an object that consists of
// a string rep and an array of alternating node specs and ops.
Toc.prototype.compileRule = function(ruleString) {
  var ruleArray = this.parseRule(ruleString).reverse();
  var rule = new Object();
  rule.array = new Array();
  rule.str = "";
  for (var i=0; i<ruleArray.length; ++i) {
    if (i % 2 == 0) {
      rule.array.push(this.parseNodeSpec(ruleArray[i]));
      var decorStr = "";
      var decorValue = "";
      if (rule.array[i].decor != null) {
	decorValue = rule.array[i].decorValue;
	switch(rule.array[i].decor) {
	case '.':
	  decorStr = "-Class-";
	  break;
	case '#':
	  if (decorValue == '-')
	    decorValue = "";
	  else
	    decorStr = "-Id-";
	  break;
	}
      }
      rule.str = rule.array[i].tag + decorStr + decorValue + rule.str;
    } else {
      rule.array.push(ruleArray[i]);
      var opStr = "";
      switch(ruleArray[i]) {
      case(' '):
        opStr = "-Ancestor-";
        break;
      case('>'):
        opStr = "-Child-";
        break;
      case('+'):
        opStr = "-Sibling-";
        break;
      }
      rule.str = opStr + rule.str;
    }
  }
  rule.str = "TOC-" + rule.str;
  return rule;
}

// Compile a tocSpec
Toc.prototype.compileTocSpec = function(tocSpec) {
  // Extract toc title
  var tmp = tocSpec.split(/\:/);
  this.tocTitle = tmp[0];

  // Extract rule list
  var ruleStrings = tmp[1].split(/,/g);

  // Compile rules
  this.ruleList = new Array(ruleStrings.length);
  for (i=0; i< ruleStrings.length; ++i) {
    this.ruleList[i] = this.compileRule(ruleStrings[i]);
  }
}

// Return true if node matches spec, false otherwise.
Toc.prototype.matchRule = function(node, spec) {
  if (node.nodeName == spec.tag) {
      var attrValue;
    switch (spec.decor) {
      // Rule expects no decoration.
      case null:
// Omit check of presence of id because an id may be used to create an anchor
// and we don't want to miss TOCing elements with id anchors.
// 	attrValue = getAttrValue("id");
// 	if (attrValue != null)
// 	  return false;
	attrValue = getAttrValue(node, "class");
	if (attrValue != null)
	  return false;
	break;
      // Rule expects id
      case '#':
	attrValue = getAttrValue(node, "id");
	if ((attrValue == null) && (spec.decorValue == '-'))
	  return true;
	if ((attrValue == null) || (attrValue != spec.decorValue))
	  return false;
	break;
      // Rule expects class
      case '.':
	attrValue = getAttrValue(node, "class");
	if ((attrValue == null) || (attrValue != spec.decorValue))
	  return false;
	break;
    }
    return true;
  } else
    return false;
}

// Toc the given node if it matches the given rule. Return true for a
// match, false otherwise.
Toc.prototype.tocWithRule = function(node, rule) {
  var currentNode = node
  if (!this.matchRule(currentNode, rule.array[0]))
    return false;
  var i = 1;
  while (i<rule.array.length) {
    var op = rule.array[i];
    ++i;			// Next node.
    switch (op) {
    case(' '):
      while (true) {
        currentNode = currentNode.parentNode;
	if (currentNode == null)
	  return false;
        if (this.matchRule(currentNode, rule.array[i]))
	  break;
      }
      break;
    case('>'):
      currentNode = currentNode.parentNode;
      if (!this.matchRule(currentNode, rule.array[i]))
	return false;
      break;
    case('+'):
      for (;;) {
	currentNode = currentNode.previousSibling;
	if (currentNode == null)
	  return false;
	if (currentNode.nodeType == 1)
	  break;
      }
      if (!this.matchRule(currentNode, rule.array[i]))
        return false;
      break;
    }
    ++i;			// Next op.
  }
  this.addEntry(node, rule.str);
  return true;
}

// Toc node if it matches one of the rules.
Toc.prototype.tocThis = function(node) {
  for (var i=0; i<this.ruleList.length; ++i)
    if (this.tocWithRule(node, this.ruleList[i]))
      break;
}

// Build a toc
Toc.prototype.build = function(tocId, tocClass) {
  // Set default arguments.
  if (arguments.length == 0)
    tocId = "begin-toc";	// Backwards compatibility with old pages.
  if (arguments.length < 2)
    tocClass = "toc";
  
  // Abort if <a class=tocId> is missing or has empty content
  var beginToc = document.getElementById(tocId);
  if (beginToc == null || beginToc.firstChild == null)
    return;

  // beginToc node should not be displayed!
  beginToc.style.display = "none";

  // Mark end of toc search
  var endId = tocId + "-end";
  document.write("<a id='" + endId + "'></a>");
  var endElement = document.getElementById(endId);

  // Extract content (a 'toc spec') of <a class=tocId> element.
  var tocSpec = beginToc.firstChild.nodeValue;

  // Compile toc spec producing a toc title (this.tocTitle) and rules
  // (this.nodeSpecList and this.opList)
  this.compileTocSpec(tocSpec);

  // Create container for toc body.
  this.tocBody = document.createElement("DIV");
  setClassAttribute(this.tocBody, "toc-body");

  // Visit all nodes in toc region and see if they match the toc
  // rules.
  this.idCount = 0;
  this.tocPrefix = tocId;
  var visitor = new Visitor(beginToc, endElement);
  var node;
  while ((node = visitor.next()) != null)
    if (node.nodeType == 1)
      this.tocThis(node);

  // Create outer container for toc.  Outer container holds
  // toc title and toc body.
  var tocContainer = document.createElement("DIV");
  setClassAttribute(tocContainer, tocClass);

  // Create element for toc title.
  var tocTitleElement = document.createElement("H1");
//  setClassAttribute(tocTitleElement, "toc-title");
  tocTitleElement.appendChild(document.createTextNode(this.tocTitle));

  // Stitch everything together.
  tocContainer.insertBefore(tocTitleElement, null);
  tocContainer.insertBefore(this.tocBody, null);
  beginToc.parentNode.insertBefore(tocContainer, beginToc.nextSibling);
}

