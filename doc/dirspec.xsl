<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>

<xsl:template match="/dir">
<xsl:processing-instruction name="cocoon-format">type="text/html"</xsl:processing-instruction>
<html>
<head>
<title>SCIRun/<xsl:value-of select="@name" /></title>
<link rel="stylesheet" type="text/css">
<xsl:attribute name="href">
<xsl:value-of select="concat($dir,'/doc/doc_styles.css')" />
</xsl:attribute>
</link>
</head>
<body>
<!-- THE SCI LOGO -->
<center><img src="../../../../doc/images/research_header_sm.gif"
usemap="#head-links" height="46" width="600" border="0" /></center>
<map name="head-links">
<area shape="rect" coords="491,15,567,32"
href="http://www.sci.utah.edu/research/research.html" />
<area shape="rect" coords="31,9,95,36" href="http://www.sci.utah.edu" />
</map>
<p class="title">
<xsl:value-of select="@name" />
</p>

<p class="head">Fit of <xsl:value-of select="@name" /></p>

<xsl:for-each select="/dir/fit/p">
	<p class="nextpara"><xsl:value-of select="." /></p> 
</xsl:for-each>

<p class="head">Use of <xsl:value-of select="@name" /></p>

<p class="subhead">Why</p>
<xsl:for-each select="/dir/use/why/p">
	<p class="nextpara"><xsl:value-of select="." /></p> 
</xsl:for-each>

<p class="subhead">When</p>
<xsl:for-each select="/dir/use/when/p">
	<p class="nextpara"><xsl:value-of select="." /></p> 
</xsl:for-each>

<p class="head">Definition</p>
<xsl:value-of select="/dir/use/definition/term" />
<p class="subhead">Examples</p>
<xsl:for-each select="/dir/use/definition/p">
	<p class="nextpara"><xsl:value-of select="." /></p> 
</xsl:for-each>




<center>
<hr />
<font face="arial, helvetica, sans-serif" size="-2">&#169; 2001, Scientific Computing and Imaging Institute &#149; <a href="http://www.utah.edu">University of Utah</a></font>
</center>

</body>
</html>
</xsl:template>
</xsl:stylesheet>
