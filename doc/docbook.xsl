<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>
<xsl:param name="cont"/>

<xsl:template match="chapter/title">
  <p class="title"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="chapter/subtitle">
  <p class="subtitle"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="sect1/title">
  <p class="head"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="sect2/title">
  <p class="subhead"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="para">
  <p class="firstpara"><xsl:apply-templates/></p>
</xsl:template>

<xsl:template match="itemizedlist">
  <ul><xsl:apply-templates/></ul>
</xsl:template>

<xsl:template match="orderedlist">
  <ol><xsl:apply-templates/></ol>
</xsl:template>

<xsl:template match="listitem">
  <li><xsl:apply-templates/></li>
</xsl:template>

<xsl:template match="citetitle">
  <b><i><xsl:value-of select="."/></i></b>
</xsl:template>

<xsl:template match="computeroutput">
  <div class="box">
    <font color="blue">
      <pre class="example"><xsl:apply-templates/></pre>
    </font>
  </div>
</xsl:template>

<xsl:template match="term">
  <b><i><font color="darkgreen"><xsl:value-of select="."/></font></i></b>
</xsl:template>

<xsl:template match="emphasis">
  <b><xsl:apply-templates/></b>
</xsl:template>

<xsl:template match="ulink">
  <a>
    <xsl:attribute name="href">
      <xsl:value-of select="@url"/>
    </xsl:attribute>
    <xsl:apply-templates/>
  </a>
</xsl:template>

<xsl:template match="/book">
<xsl:processing-instruction name="cocoon-format">type="text/html"</xsl:processing-instruction>
<html>

<xsl:variable name="swidk">
<xsl:choose>
  <xsl:when test="$dir=4">../../../..</xsl:when>
  <xsl:when test="$dir=3">../../..</xsl:when>
  <xsl:when test="$dir=2">../..</xsl:when>
  <xsl:when test="$dir=1">..</xsl:when>
</xsl:choose>
</xsl:variable>

<head>
<title><xsl:value-of select="./bookinfo/title" /></title>
<link rel="stylesheet" type="text/css">
<xsl:attribute name="href">
<xsl:value-of select="concat($swidk,'/doc/doc_styles.css')" />
</xsl:attribute>
</link>

</head>
<body>

<!-- *************************************************************** -->
<!-- *************** STANDARD SCI RESEARCH HEADER ****************** -->
<!-- *************************************************************** -->

<center>
<img usemap="#head-links" height="71" width="600" border="0">
<xsl:attribute name="src">
<xsl:value-of select="concat($swidk,'/doc/images/research_menuheader.jpg')" />
</xsl:attribute>
</img>
</center>
<map name="head-links">
	<area href="http://www.sci.utah.edu" shape="rect" coords="7,4,171,33" alt="SCI Home" />
	<area href="http://www.sci.utah.edu/software" shape="rect" coords="490,10,586,32" alt="Software" />

	<area shape="rect" coords="340,10,480,32" alt="Documentation">
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/index.html')" />
        </xsl:attribute>
        </area>
 

        <area coords="0,41,156,64" shape="rect" alt="Installation Guide">
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/InstallGuide/installguide.html')" />
        </xsl:attribute>
        </area>

 
        <area coords="157,41,256,64" shape="rect" alt="User's Guide"> 
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/UserGuide/userguide.html')" />
        </xsl:attribute>
        </area>

        <area coords="257,41,397,64" shape="rect" alt="Developer's Guide">
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/DeveloperGuide/devguide.html')" />
        </xsl:attribute>
        </area>
 
        <area coords="398,41,535,64" shape="rect" alt="Reference Guide">  
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/ReferenceGuide/refguide.html')" />
        </xsl:attribute>
        </area>

        <area coords="536,41,600,64" shape="rect" alt="FAQ">  
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/FAQ/faq.html')" />
        </xsl:attribute>
        </area>
</map> 

<!-- *************************************************************** -->
<!-- *************************************************************** -->

<!-- ******* identify *this* source document -->
<xsl:variable name="source">
  <xsl:value-of select="/book/bookinfo/title"/>
</xsl:variable>

<xsl:if test="$cont=0">

<!-- ********** Table of Contents ********* -->

<p class="title">
  <xsl:value-of select="./title" />
</p>

<p class="subtitle">
  <xsl:value-of select="./subtitle" />
</p>

<hr size="1"/>

<xsl:for-each select="./preface">
  <xsl:for-each select="./para">
    <p class="firstpara"><xsl:value-of select="." /></p>
  </xsl:for-each>
</xsl:for-each>

<hr size="1"/>

<xsl:for-each select="./chapter">
  <xsl:variable name="chapnum"><xsl:number/></xsl:variable>
  <p class="head">
    <a>
      <xsl:attribute name="href">
        <xsl:value-of select="concat($source,'?dir=2&amp;cont=',$chapnum)"/>
      </xsl:attribute>  
      <xsl:value-of select="./title" />
    </a>
  </p>
  <p class="firstpara">
    <xsl:value-of select="./sect1/para" />
  </p>
</xsl:for-each>

</xsl:if>

<xsl:if test="$cont>0">

<!-- *********** Chapters ************ -->

<xsl:for-each select="./chapter">
  <xsl:variable name="chapnum"><xsl:number/></xsl:variable>
  <xsl:if test="$chapnum=$cont">
    <xsl:apply-templates/>
  </xsl:if>
</xsl:for-each>

</xsl:if>

<!-- ******************************************************************* -->
<!-- *********************** STANDARD SCI FOOTER *********************** -->
<!-- ******************************************************************* -->
<center>
<hr size="1" width="600" />
<font size="-1"><a href="http://www.sci.utah.edu">Scientific Computing and Imaging Institute</a> &#149; <a href="http://www.utah.edu">University of Utah</a> &#149; 
(801) 585-1867</font>
</center>
<!-- ********************* END STANDARD SCI FOOTER ********************* -->
<!-- ******************************************************************* -->

</body>
</html>
</xsl:template>

</xsl:stylesheet>
