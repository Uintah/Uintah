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
<xsl:param name="dir"/>
<xsl:param name="cont"/>

<!-- ************************************************************ -->
<!-- ***************** web displayable templates **************** -->
<!-- ************************************************************ -->

<xsl:template match="beginpage" mode="web">
</xsl:template>

<xsl:template match="table" mode="web">
  <center>
    <table border="1"><xsl:apply-templates mode="web"/></table>
  </center>
</xsl:template>

<xsl:template match="entrytbl" mode="web">
  <table border="0"><xsl:apply-templates mode="web"/></table>
</xsl:template>

<xsl:template match="tgroup" mode="web">
  <xsl:apply-templates mode="web"/>
</xsl:template>

<xsl:template match="thead" mode="web">
  <xsl:for-each select="./row">
    <tr>
      <xsl:for-each select="./entry">
        <th class="firstpara"><xsl:apply-templates mode="web" /></th>
      </xsl:for-each>
    </tr>
  </xsl:for-each>
</xsl:template>

<xsl:template match="tbody" mode="web">
  <xsl:apply-templates mode="web"/>
</xsl:template>

<xsl:template match="row" mode="web">
  <tr><xsl:apply-templates mode="web"/></tr>
</xsl:template>

<xsl:template match="entry" mode="web">
  <td class="firstpara"><xsl:apply-templates mode="web"/></td>
</xsl:template>

<xsl:template match="chapter/title" mode="web">
  <p class="title"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="chapter/subtitle" mode="web">
  <p class="subtitle"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="sect1/title" mode="web">
  <xsl:param name="sectn"/>
  <a>
    <xsl:attribute name="name">
      <xsl:value-of select="$sectn"/>
    </xsl:attribute>
    <p class="head"><xsl:value-of select="."/></p>
  </a>
</xsl:template>

<xsl:template match="sect2/title" mode="web">
  <p class="subhead"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="para" mode="web">
  <p class="firstpara"><xsl:apply-templates mode="web"/></p>
</xsl:template>

<xsl:template match="itemizedlist" mode="web">
  <ul class="list"><xsl:apply-templates mode="web"/></ul>
</xsl:template>

<xsl:template match="orderedlist" mode="web">
  <ol><xsl:apply-templates mode="web"/></ol>
</xsl:template>

<xsl:template match="listitem" mode="web">
  <li><xsl:apply-templates mode="web"/></li>
</xsl:template>

<xsl:template match="citetitle" mode="web">
  <b><i><xsl:value-of select="."/></i></b>
</xsl:template>

<xsl:template match="computeroutput" mode="web">
  <div class="box"><br/>
    <font color="blue">
      <pre class="example"><xsl:apply-templates mode="web"/></pre>
    </font>
  </div>
</xsl:template>

<xsl:template match="term" mode="web">
  <b><i><font color="darkgreen"><xsl:value-of select="."/></font></i></b>
</xsl:template>

<xsl:template match="emphasis" mode="web">
  <b><xsl:apply-templates mode="web"/></b>
</xsl:template>

<xsl:template match="ulink" mode="web">
  <a>
    <xsl:attribute name="href">
      <xsl:value-of select="@url"/>
    </xsl:attribute>
    <xsl:apply-templates mode="web"/>
  </a>
</xsl:template>


<!-- ************************************************************ -->
<!-- *********************** printable templates **************** -->
<!-- ************************************************************ -->

<xsl:template match="beginpage" mode="print">
  <hr size="3" />
</xsl:template>

<xsl:template match="table" mode="print">
  <center>
    <table border="1"><xsl:apply-templates mode="print"/></table>
  </center>
</xsl:template>

<xsl:template match="entrytbl" mode="print">
  <table border="0"><xsl:apply-templates mode="print"/></table>
</xsl:template>

<xsl:template match="tgroup" mode="print">
  <xsl:apply-templates mode="print"/>
</xsl:template>

<xsl:template match="thead" mode="print">
  <xsl:for-each select="./row">
    <tr>
      <xsl:for-each select="./entry">
        <th class="pfirstpara"><xsl:apply-templates mode="print" /></th>
      </xsl:for-each>
    </tr>
  </xsl:for-each>
</xsl:template>

<xsl:template match="tbody" mode="print">
  <xsl:apply-templates mode="print"/>
</xsl:template>

<xsl:template match="row" mode="print">
  <tr><xsl:apply-templates mode="print"/></tr>
</xsl:template>

<xsl:template match="entry" mode="print">
  <td class="pfirstpara"><xsl:apply-templates mode="print"/></td>
</xsl:template>

<xsl:template match="chapter/title" mode="print">
  <p class="ptitle"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="chapter/subtitle" mode="print">
  <p class="psubtitle"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="sect1/title" mode="print">
  <p class="phead"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="sect2/title" mode="print">
  <p class="psubhead"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="para" mode="print">
  <p class="pfirstpara"><xsl:apply-templates mode="print"/></p>
</xsl:template>

<xsl:template match="itemizedlist" mode="print">
  <ul class="plist"><xsl:apply-templates mode="print"/></ul>
</xsl:template>

<xsl:template match="orderedlist" mode="print">
  <ol><xsl:apply-templates mode="print"/></ol>
</xsl:template>

<xsl:template match="listitem" mode="print">
  <li><xsl:apply-templates mode="print"/></li>
</xsl:template>

<xsl:template match="citetitle" mode="print">
  <b><i><xsl:value-of select="."/></i></b>
</xsl:template>

<xsl:template match="computeroutput" mode="print">
  <div class="box"><br/>
    <font color="blue">
      <pre class="pexample"><xsl:apply-templates mode="print"/></pre>
    </font>
  </div>
</xsl:template>

<xsl:template match="term" mode="print">
  <b><i><font color="darkgreen"><xsl:value-of select="."/></font></i></b>
</xsl:template>

<xsl:template match="emphasis" mode="print">
  <b><xsl:apply-templates mode="print"/></b>
</xsl:template>

<xsl:template match="ulink" mode="print">
  <xsl:apply-templates mode="print"/>
</xsl:template>

<xsl:template match="/book">
<xsl:processing-instruction name="cocoon-format">type="text/html"</xsl:processing-instruction>

<xsl:variable name="swidk">
<xsl:choose>
  <xsl:when test="$dir=4">../../../..</xsl:when>
  <xsl:when test="$dir=3">../../..</xsl:when>
  <xsl:when test="$dir=2">../..</xsl:when>
  <xsl:when test="$dir=1">..</xsl:when>
</xsl:choose>
</xsl:variable>
<html>

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

<xsl:if test="$cont!='print'">

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

</xsl:if>

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
  <xsl:apply-templates mode="web"/>
</xsl:for-each>

<hr size="1"/>

<xsl:for-each select="./chapter">
  <xsl:variable name="chapnum"><xsl:number/></xsl:variable>

  <p class="head">
    <xsl:value-of select="$chapnum"/>
    <xsl:value-of select="concat(' ',' ')"/>
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
    <p class="title">Chapter <xsl:value-of select="$chapnum"/>: <xsl:value-of select="./title"/></p>
  </xsl:if>

  <xsl:if test="$chapnum=$cont">
    <xsl:for-each select="./sect1">
      <xsl:variable name="sectnum"><xsl:number/></xsl:variable>
      <xsl:apply-templates mode="web">
        <xsl:with-param name="sectn">
          <xsl:value-of select="$sectnum"/>
        </xsl:with-param>
      </xsl:apply-templates>
    </xsl:for-each>
  </xsl:if>

</xsl:for-each>

</xsl:if>

<xsl:if test="$cont='print'">

<!-- ************** Print all ************** -->

<p class="ptitle">
  <xsl:value-of select="./title" />
</p>

<p class="psubtitle">
  <xsl:value-of select="./subtitle" />
</p>

<hr size="1"/>

<xsl:for-each select="./preface">
  <xsl:apply-templates mode="print"/>
</xsl:for-each>

<hr size="1"/>

<xsl:for-each select="./chapter">
  <xsl:variable name="chapnum"><xsl:number/></xsl:variable>

  <p class="phead">
    <xsl:value-of select="$chapnum"/> 
    <xsl:value-of select="concat(' ',' ')"/>
    <xsl:value-of select="./title" />
  </p>

  <p class="pfirstpara">
    <xsl:value-of select="./sect1/para" />
  </p>

</xsl:for-each>

<xsl:for-each select="./chapter">
  <xsl:variable name="chapnum"><xsl:number/></xsl:variable>

  <p class="ptitle">Chapter <xsl:value-of select="$chapnum"/>: <xsl:value-of select="./title"/></p>

  <xsl:for-each select="./sect1">
    <xsl:variable name="sectnum"><xsl:number/></xsl:variable>
    <xsl:apply-templates mode="print">
      <xsl:with-param name="sectn">
        <xsl:value-of select="$sectnum"/>
      </xsl:with-param>
    </xsl:apply-templates>
  </xsl:for-each>

</xsl:for-each>

</xsl:if>

<!-- ******************************************************************* -->
<!-- *********************** STANDARD SCI FOOTER *********************** -->
<!-- ******************************************************************* -->

<xsl:if test="$cont!='print'">

<center>
<hr size="1" width="600" />
<font size="-1"><a href="http://www.sci.utah.edu">Scientific Computing and Imaging Institute</a> &#149; <a href="http://www.utah.edu">University of Utah</a> &#149; 
(801) 585-1867</font>
</center>

</xsl:if>

<!-- ********************* END STANDARD SCI FOOTER ********************* -->
<!-- ******************************************************************* -->

</body>
</html>
</xsl:template>

</xsl:stylesheet>
