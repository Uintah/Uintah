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

<!-- Override docbook templates.  See srdocbook-common.xsl for details.  -->

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
version="1.0"> 

<xsl:import href="chunk.xsl"/>

<xsl:include href="srdocbook-common.xsl"/>

<!-- When chunking, put the user header content above the top navigators
and put the user footer content below the bottom navigators. -->

<xsl:template name="chunk-element-content">
  <xsl:param name="prev"></xsl:param>
  <xsl:param name="next"></xsl:param>

  <html>
    <xsl:call-template name="html.head">
      <xsl:with-param name="prev" select="$prev"/>
      <xsl:with-param name="next" select="$next"/>
    </xsl:call-template>

    <body>
      <xsl:call-template name="body.attributes"/>
      <xsl:call-template name="user.header.content"/>

      <xsl:call-template name="user.header.navigation"/>

      <xsl:call-template name="header.navigation">
        <xsl:with-param name="prev" select="$prev"/>
        <xsl:with-param name="next" select="$next"/>
      </xsl:call-template>
      
      <xsl:apply-imports/>

      <xsl:call-template name="footer.navigation">
        <xsl:with-param name="prev" select="$prev"/>
        <xsl:with-param name="next" select="$next"/>
      </xsl:call-template>

      <xsl:call-template name="user.footer.navigation"/>

      <xsl:call-template name="user.footer.content"/>
      
    </body>
  </html>
</xsl:template>

</xsl:stylesheet>
