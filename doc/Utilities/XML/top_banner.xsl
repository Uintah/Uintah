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

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template name="top_banner">

<div class="top-banner-margins">
<table border="0" cellspacing="0" cellpadding="0" width="100%" height="91">
<tr>
<td align="center" width="%100">
<xsl:attribute name="background">
  <xsl:value-of select="concat($treetop, '/doc/Utilities/Figures/banner_top_fill.jpg')"/>
</xsl:attribute>
<img width="744" height="91" border="0" usemap="#banner">
<xsl:attribute name="src">
  <xsl:value-of select="concat($treetop, '/doc/Utilities/Figures/banner_top.jpg')"/>
</xsl:attribute>
</img>
</td>
<map name="banner">
<area href="http://www.sci.utah.edu" alt="Home" coords="92,62,186,83" />
<area href="http://software.sci.utah.edu" alt="Software" coords="193,61,289,83" />

<area coords="296,62,437,83">
<xsl:attribute name="href">
  <xsl:value-of select="concat($treetop,'/doc/index.html')" />
</xsl:attribute>
</area>

<area coords="449,62,544,83">
<xsl:attribute name="href">
<xsl:value-of select="concat($treetop,'/doc/User/Guide/usersguide/index.html')" />
</xsl:attribute>
</area>

<area coords="550,62,692,83">
<xsl:attribute name="href">
<xsl:value-of select="concat($treetop,'/doc/Developer/Guide/TOC.html')" />
</xsl:attribute>
</area>

<area coords="550,62,692,83">
<xsl:attribute name="href">
<xsl:value-of select="concat($treetop,'/doc/Installation/Guide/TOC.html')" />
</xsl:attribute>
</area>

</map>
</tr>
</table>
</div>

</xsl:template>

</xsl:stylesheet>
