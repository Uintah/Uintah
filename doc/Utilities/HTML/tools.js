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

var gTreeTop = findTreeTop();

function findTreeTop() {
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
  document.write('<img class="top-banner" src="', gTreeTop, 'doc/Utilities/Figures/doc_banner04.jpg" border="0" usemap="#banner"> \
<map name="banner">\
<area href="http://www.sci.utah.edu" coords="133,103,212,124" alt="SCI Home">\
<area href="http://software.sci.utah.edu" coords="213,103,296,124" alt="Software">\
<area href="', gTreeTop, 'doc/" coords="297,103,420,124" alt="Documentation">\
<area href="', gTreeTop, 'doc/Installation" coords="421,103,524,124" alt="Installation">\
<area href="', gTreeTop, 'doc/User/" coords="525,103,571,124" alt="User">\
<area href="', gTreeTop, 'doc/Developer/" coords="572,103,667,124" alt="Developer">\
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

