<?xml version="1.0"?>

<!--
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
-->

<!-- The following templates override those found in docbook.xsl and
     docbook-chunk.xsl --> 

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
version="1.0">

  <!-- Generate java script code that locates root of doc tree -->
  <xsl:template name="user.head.content">
    <script language="JavaScript">
      var treetop="";
      var path = location.pathname;
      while (path.substr(path.lastIndexOf("/")+1) != "doc") {
      treetop += "../";
      path = path.substr(0, path.lastIndexOf("/"));
      }
      document.write("&lt;link href='",treetop,"doc/Utilities/HTML/srdocbook.css' rel='stylesheet' type='text/css'&gt;")
    </script>
  </xsl:template>

  <!-- Generate SCI header banner -->
  <xsl:template name="user.header.content">
    <script language="JavaScript">
      document.write('&lt;script language="JavaScript" src="',treetop,'doc/Utilities/HTML/banner_top.js"&gt;&lt;\/script&gt;');
    </script>
  </xsl:template>

  <!-- Generate SCI footer. -->
  <xsl:template name="user.footer.content">
    <script language="JavaScript">
      document.write('&lt;script language="JavaScript" src="',treetop,'doc/Utilities/HTML/banner_bottom.js"&gt;&lt;\/script&gt;');
    </script>
  </xsl:template>

</xsl:stylesheet>
