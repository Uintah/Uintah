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

<xsl:template name="bottom_banner">

<div class="bottom-banner-margins">
<table border="0" cellspacing="0" cellpadding="0" height="32" width="100%">
<tr>
<td align="left" width="%100">
<xsl:attribute name="background">
<xsl:value-of select="concat($treetop, 'doc/Utilities/Figures/banner_bottom_fill.jpg')"/>
</xsl:attribute>
<img width="444" height="32" border="0">
<xsl:attribute name="src">
<xsl:value-of select="concat($treetop, 'doc/Utilities/Figures/banner_bottom.jpg')"/>
</xsl:attribute>
</img>
</td>
</tr>
</table>
</div>
<center><font size="-2" face="arial, helvetica, sans-serif">Scientific
Computing and Imaging Institute &#149; 50 S. Central Campus Dr. Rm
3490 &#149; Salt Lake City, UT 84112<br />

(801) 585-1867 &#149; fax: (801) 585-6513 &#149; <a href="http://www.utah.edu/disclaimer/disclaimer_home.html">Disclaimer</a></font></center>


</xsl:template>

</xsl:stylesheet>
