<?xml version="1.0"?> <!-- -*- nxml -*- -->

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

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

  <!-- treetop is relative path to top of sr tree -->
  <xsl:param name="treetop"/>

  <!-- Include customizations of qandaset -->
  <xsl:include href="srqandaset.xsl"/>

  <!-- Generate java script code that locates root of doc tree -->
  <xsl:template name="user.head.content">
    <link rel="stylesheet" type="text/css">
      <xsl:attribute name="href">
	<xsl:value-of select="$treetop"/>/doc/Utilities/HTML/srdocbook.css<xsl:text/>
      </xsl:attribute>
    </link>
  </xsl:template>

  <!-- Generate page "pre-content" -->
  <xsl:template name="user.header.content">
    <script type="text/javascript">
      <xsl:attribute name="src">
	<xsl:value-of select="$treetop"/>/doc/Utilities/HTML/tools.js<xsl:text/>
      </xsl:attribute>
    </script>
    <script type="text/javascript">preDBContent();</script>
  </xsl:template>

  <!-- Generate page "post-content" -->
  <xsl:template name="user.footer.content">
    <script type="text/javascript">postDBContent(); </script>
  </xsl:template>

  <!-- Change type from a 'charseq' to a 'monoseq' -->
  <xsl:template match="type">
    <xsl:call-template name="inline.monoseq"/>
  </xsl:template>

</xsl:stylesheet>
