<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>

<xsl:template match="developer" mode="dev">
  <xsl:apply-templates/>
</xsl:template>

<xsl:template match="orderedlist">
  <ol><xsl:apply-templates/></ol>
</xsl:template>

<xsl:template match="unorderedlist">
  <ul><xsl:apply-templates/></ul>
</xsl:template>

<xsl:template match="desclist">
  <ul><p class="firstpara"><xsl:apply-templates/></p></ul>
</xsl:template>

<xsl:template match="term">
  <b><i><font color="green">
    <xsl:value-of select="."/>
  </font></i></b>
</xsl:template>

<xsl:template match="keyboard">
  <div class="box"><pre class="example"><font color="blue">
    <xsl:apply-templates/>
  </font></pre></div>
</xsl:template>

<xsl:template match="cite">
  <b><i><xsl:apply-templates/></i></b>
</xsl:template>

<xsl:template match="rlink">
  <a>
    <xsl:attribute name="href">
      <xsl:value-of select="@path"/>
    </xsl:attribute>
    <xsl:apply-templates/>
  </a>
</xsl:template>

<xsl:template match="examplesr">
  <a>
    <xsl:attribute name="href">
      <xsl:value-of select="."/>
    </xsl:attribute>
    <p class="subtitle">
      Example Network
    </p>
  </a>
</xsl:template>

<xsl:template match="developer">
</xsl:template>

<xsl:template match="parameter">
  <tr>
    <td><xsl:value-of select="./widget"/></td>
    <td><xsl:value-of select="./label"/></td>
    <td><xsl:value-of select="./description"/></td>
  </tr>
</xsl:template>

<xsl:template match="img">
  <center>
    <img vspace="20">
      <xsl:attribute name="src">
        <xsl:value-of select="."/>
      </xsl:attribute>
    </img>
  </center>
</xsl:template>

<xsl:template match="port">
  <xsl:param name="last"/>
  <tr>
    <td>
      <xsl:choose>
        <xsl:when test="$last='last'">Dynamic Port</xsl:when>
        <xsl:otherwise>Port</xsl:otherwise>
      </xsl:choose>
    </td>
    <td><xsl:value-of select="./datatype"/></td>
    <td><xsl:value-of select="./name"/></td>
    <td><xsl:value-of select="./description"/></td>
  </tr>
</xsl:template>

<xsl:template match="file">
  <tr>
    <td>File</td>
    <td><xsl:value-of select="./datatype"/></td>
    <td></td>
    <td><xsl:value-of select="./description"/></td>
  </tr>
  
</xsl:template>

<xsl:template match="device">
  <tr>
    <td>Device</td>
    <td></td>
    <td><xsl:value-of select="./devicename"/></td>
    <td><xsl:value-of select="./description"/></td>
  </tr>
</xsl:template>

<xsl:template match="p">
  <p class="firstpara"><xsl:apply-templates/></p>
</xsl:template>

<xsl:template match="/component">
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
<title><xsl:value-of select="@name" /></title>
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
        <xsl:value-of select="concat($swidk,'/doc/InstallGuide/installguide.xml?cont=0&amp;dir=2')" />
        </xsl:attribute>
        </area>

 
        <area coords="157,41,256,64" shape="rect" alt="User's Guide"> 
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/UserGuide/userguide.html')" />
        </xsl:attribute>
        </area>

        <area coords="257,41,397,64" shape="rect" alt="Developer's Guide">
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/DeveloperGuide/devguide.xml?cont=0&amp;dir=2')" />
        </xsl:attribute>
        </area>
 
        <area coords="398,41,535,64" shape="rect" alt="Reference Guide">  
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/ReferenceGuide/refguide.html')" />
        </xsl:attribute>
        </area>

        <area coords="536,41,600,64" shape="rect" alt="FAQ">  
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/FAQ/faq.xml?cont=0&amp;dir=2')" />
        </xsl:attribute>
        </area>
</map> 

<!-- *************************************************************** -->
<!-- *************************************************************** -->

<p class="title"><xsl:value-of select="@name"/></p>
<p class="subtitle">Category: <xsl:value-of select="@category"/></p>

<xsl:apply-templates select="./overview/examplesr"/>

<p class="head">Summary</p>

<p class="firstpara">
  <xsl:for-each select="./overview/summary">
    <xsl:apply-templates/>
  </xsl:for-each>
</p>

<p class="head">Description</p>

<p class="firstpara">
  <xsl:for-each select="./overview/description">
    <xsl:apply-templates/>
  </xsl:for-each>
</p>

<p class="head">I/O</p>

<p class="firstpara">
<table border="1">
  <tr>
    <th align="left">I/O Type</th>
    <th align="left">Datatype</th>
    <th align="left">Name</th>
    <th align="left">Description</th>
  </tr>
    <xsl:for-each select="./io">
      <xsl:for-each select="./inputs">
        <tr><th colspan="5" align="left">Inputs</th></tr>
        <xsl:variable name="dynamic">
          <xsl:value-of select="@lastportdynamic"/>
        </xsl:variable> 
        <xsl:for-each select="./port">
          <xsl:variable name="portnum"><xsl:number/></xsl:variable>
          <xsl:variable name="last">
            <xsl:if test="$portnum=last() and $dynamic='yes'">last</xsl:if>
          </xsl:variable>
          <xsl:apply-templates select=".">
            <xsl:with-param name="last">
              <xsl:value-of select="$last"/>
            </xsl:with-param>
          </xsl:apply-templates>
        </xsl:for-each>
        <xsl:for-each select="./file">
          <xsl:apply-templates select="."/>
        </xsl:for-each>
        <xsl:for-each select="./device">
          <xsl:apply-templates select="."/>
        </xsl:for-each>
      </xsl:for-each>
      <xsl:for-each select="./outputs">
        <tr><th colspan="5" align="left">Outputs</th></tr>
        <xsl:for-each select="./port">
          <xsl:apply-templates select="."/>
        </xsl:for-each>
        <xsl:for-each select="./file">
          <xsl:apply-templates select="."/>
        </xsl:for-each>
        <xsl:for-each select="./device">
          <xsl:apply-templates select="."/>
        </xsl:for-each>
      </xsl:for-each>
    </xsl:for-each>
</table>
</p>

<xsl:for-each select="./gui">
  <p class="head">GUI</p>
  <xsl:apply-templates select="./description"/>
  <xsl:apply-templates select="./img"/>
  <p class="firstpara">
  <table border="1">
    <tr><th>Widget</th><th>Label</th><th>Description</th></tr>
    <xsl:for-each select="./parameter">
      <xsl:apply-templates select="."/>
    </xsl:for-each>
  </table>
  </p>
</xsl:for-each>


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
