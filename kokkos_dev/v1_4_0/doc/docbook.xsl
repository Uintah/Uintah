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

<xsl:template name="figcount">
<xsl:param name="fignode"/>
<xsl:for-each select="$fignode">
<xsl:if test="position()=1">
<xsl:value-of select="count(ancestor|preceding::figure) + 1"/>
</xsl:if>
</xsl:for-each>
</xsl:template>

<xsl:template match="fignum" mode="web">
<xsl:variable name="figname">
<xsl:value-of select="@name"/>
</xsl:variable>
<xsl:call-template name="figcount">
<xsl:with-param name="fignode" select="//figure[@name=$figname]"/>
</xsl:call-template>
</xsl:template>

<xsl:template match="figure" mode="web">
<xsl:variable name="fignum">
<xsl:value-of select="count(ancestor|preceding::figure) + 1"/>
</xsl:variable>
<center><table border="0">
<tr>
<xsl:for-each select="@*">
<xsl:if test="'img'=substring(name(.),1,3)">
<td>
<img>
<xsl:attribute name="src">
<xsl:value-of select="."/>
</xsl:attribute>
</img>
</td>
</xsl:if>
</xsl:for-each>
</tr>
<tr>
<xsl:for-each select="@*">
<xsl:variable name="subnum">
<xsl:if test="last() &lt;= 3">
</xsl:if>
<xsl:if test="last() > 3">
<xsl:value-of select="substring(name(.),4,4)"/>
</xsl:if>
</xsl:variable>
<xsl:if test="'cap'=substring(name(.),1,3)">
<td>
<p class="firstpara">
<xsl:value-of select="concat('Figure ',$fignum,$subnum,': ',.)"/>
</p>
</td>
</xsl:if>
</xsl:for-each>
</tr>
</table></center>
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

<xsl:template match="part/title" mode="web">
<p class="title"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="part/subtitle" mode="web">
<p class="subtitle"><xsl:value-of select="."/></p>
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
<div class="box">
<font color="blue">
<pre class="example">
<xsl:apply-templates mode="web"/></pre>
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

<xsl:template match="fignum" mode="print">
<xsl:variable name="figname">
<xsl:value-of select="@name"/>
</xsl:variable>
<xsl:call-template name="figcount">
<xsl:with-param name="fignode" select="//figure[@name=$figname]"/>
</xsl:call-template>
</xsl:template>

<xsl:template match="figure" mode="print">
<xsl:variable name="fignum">
<xsl:value-of select="count(ancestor|preceding::figure) + 1"/>
</xsl:variable>
<center><table border="0">
<tr>
<xsl:for-each select="@*">
<xsl:if test="'img'=substring(name(.),1,3)">
<td>
<img>
<xsl:attribute name="src">
<xsl:value-of select="."/>
</xsl:attribute>
</img>
</td>
</xsl:if>
</xsl:for-each>
</tr>
<tr>
<xsl:for-each select="@*">
<xsl:variable name="subnum">
<xsl:if test="last() &lt;= 3">
</xsl:if>
<xsl:if test="last() > 3">
<xsl:value-of select="substring(name(.),4,4)"/>
</xsl:if>
</xsl:variable>
<xsl:if test="'cap'=substring(name(.),1,3)">
<td>
<p class="pfirstpara">
<xsl:value-of select="concat('Figure ',$fignum,$subnum,': ',.)"/>
</p>
</td>
</xsl:if>
</xsl:for-each>
</tr>
</table></center>
</xsl:template>

<xsl:template match="beginpage" mode="print">
<!-- all br's translate into forced page breaks (see doc_styles.css) -->
<br/>
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

<xsl:template match="part/title" mode="print">
<p class="ptitle"><xsl:value-of select="."/></p>
</xsl:template>

<xsl:template match="part/subtitle" mode="print">
<p class="psubtitle"><xsl:value-of select="."/></p>
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
<div class="box">
<font color="blue">
<pre class="pexample">
<xsl:apply-templates mode="print"/></pre>
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
<xsl:when test="$dir=0">.</xsl:when>
</xsl:choose>
</xsl:variable>
<html>

<head>
<title><xsl:value-of select="./bookinfo/title" /></title>
<!-- Changed by: Ted Dustman, 26-Feb-2002 -->
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

<xsl:if test="$cont!='printable'">
<div class="banner-margins">
<table border="0" cellspacing="0" cellpadding="0" width="100%" height="91">
<tr>
<td align="center" width="%100">
<xsl:attribute name="background">
<xsl:value-of select="concat($swidk, '/doc/images/banner_top_fill.jpg')"/>
</xsl:attribute>
<img width="744" height="91" border="0" usemap="#banner">
<xsl:attribute name="src">
<xsl:value-of select="concat($swidk, '/doc/images/banner_top.jpg')"/>
</xsl:attribute>
</img>
</td>
<map name="banner">
<area href="http://www.sci.utah.edu" alt="Home" coords="92,62,186,83" />
<area href="http://software.sci.utah.edu" alt="Software" coords="193,61,289,83" />

<area coords="296,62,437,83">
<xsl:attribute name="href">
<xsl:value-of select="concat($swidk,'/doc/')" />
</xsl:attribute>
</area>

<area coords="449,62,544,83">
<xsl:attribute name="href">
  <xsl:value-of select="concat($swidk,'/doc/UserGuide/usersguide')" />
</xsl:attribute>
</area>

<area coords="550,62,692,83">
<xsl:attribute name="href">
  <xsl:value-of select="concat($swidk,'/doc/TechnicalGuide/TOC.html')" />
</xsl:attribute>
</area>

</map>
</tr>
</table>
</div>
</xsl:if>

<!-- *************************************************************** -->
<!-- *************************************************************** -->

<!-- ******* identify *this* source document -->
<xsl:variable name="source">
<xsl:value-of select="/book/bookinfo/title"/>
</xsl:variable>

<xsl:if test="$cont='TOC'">

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

<xsl:for-each select="./part">
<xsl:variable name="partnum"><xsl:number/></xsl:variable>

<p class="head">
<xsl:value-of select="concat('Part ',$partnum,': ',./title)"/>
</p>
<xsl:for-each select="./chapter">
<xsl:variable name="chapnum"><xsl:number/></xsl:variable>

<p class="subhead">
<xsl:value-of select="concat($chapnum,' ')"/>
<a>
<xsl:attribute name="href">
<xsl:value-of 
select="concat('p',$partnum,'c',$chapnum,'.html')"/>
</xsl:attribute>  
<xsl:value-of select="./title" />
</a>
</p>

<ul><xsl:for-each select="./sect1">
<xsl:variable name="sectnum"><xsl:number/></xsl:variable>
<xsl:if test="./title!=''">
<li>
<a>
<xsl:attribute name="href">
<xsl:value-of select="concat('p',$partnum,'c',$chapnum,
'.html#',$sectnum)"/>
</xsl:attribute> 
<xsl:value-of select="./title"/>
</a>
</li>
<ul><xsl:for-each select="./sect2">
<li><xsl:value-of select="./title"/></li>
</xsl:for-each></ul>
</xsl:if>
</xsl:for-each></ul>

</xsl:for-each>
</xsl:for-each>

</xsl:if>

<xsl:if test="$cont!='TOC' and $cont!='printable'">

<!-- *********** Chapters ************ -->

<xsl:for-each select="./part">
<xsl:variable name="partnum"><xsl:number/></xsl:variable>
<xsl:variable name="lastpart"><xsl:value-of select="last()"/></xsl:variable>

<xsl:if test="count(./chapter)=0">
<table border="0"><tr><td width="50">
PREV
</td><td>
NEXT
</td></tr></table>
</xsl:if>

<xsl:for-each select="./chapter">
<xsl:variable name="chapnum"><xsl:number/></xsl:variable>
<xsl:variable name="lastchap">
<xsl:value-of select="last()"/>
</xsl:variable>
<xsl:variable name="prev">
<xsl:value-of select="$chapnum - 1"/>
</xsl:variable>
<xsl:variable name="next">
<xsl:value-of select="$chapnum + 1"/>
</xsl:variable>

<xsl:if test="concat('p',$partnum,'c',$chapnum)=$cont">
<table border="0"><tr><td width="50">
<xsl:choose>
<xsl:when test="$chapnum&gt;1">
<a>
<xsl:attribute name="href">
<xsl:value-of
select="concat('p',$partnum,'c',$prev,'.html')"/>
</xsl:attribute>
PREV
</a>
</xsl:when>
<xsl:otherwise>
<xsl:choose>
<xsl:when test="$partnum&gt;1">
<a>
<xsl:attribute name="href">
<xsl:value-of 
select="concat('p',$partnum - 1,'c1.html')"/>
</xsl:attribute>
PREV
</a>
</xsl:when>
<xsl:otherwise>
PREV
</xsl:otherwise>
</xsl:choose>
</xsl:otherwise>
</xsl:choose>
</td>

<td width="50">
<xsl:choose>
<xsl:when test="$chapnum&lt;$lastchap">
<a>
<xsl:attribute name="href">
<xsl:value-of
select="concat('p',$partnum,'c',$next,'.html')"/>
</xsl:attribute>
NEXT
</a>
</xsl:when>
<xsl:otherwise>
<xsl:choose>
<xsl:when test="$partnum&lt;$lastpart">
<a>
<xsl:attribute name="href">
<xsl:value-of 
select="concat('p',$partnum + 1,'c1.html')"/>
</xsl:attribute>
NEXT
</a>
</xsl:when>
<xsl:otherwise>
NEXT
</xsl:otherwise>
</xsl:choose>
</xsl:otherwise>
</xsl:choose>
</td></tr></table>

<p class="title">
<xsl:value-of select="concat('Chapter ',$chapnum,': ',./title)"/>
</p>
</xsl:if>

<xsl:if test="concat('p',$partnum,'c',$chapnum)=$cont">
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
</xsl:for-each>

</xsl:if>

<xsl:if test="$cont='printable'">

<!-- ************** Print all ************** -->

<p class="title">
<xsl:value-of select="./title" />
</p>

<p class="subtitle">
<xsl:value-of select="concat('Version ',./bookinfo/edition)"/>
</p>

<center>
<img src="../images/SCI_logo.jpg" vspace="50"/>
</center>

<p class="psubtitle">
<xsl:value-of select="./subtitle" />
</p>

<hr size="1"/>

<xsl:for-each select="./preface">
<xsl:apply-templates mode="print"/>
</xsl:for-each>

<hr size="1"/>

<p class="fineprint">
<xsl:value-of select="concat('Copyright(c)',
./bookinfo/copyright/year,' ',
./bookinfo/copyright/holder)" />
</p>

<br/>

<p class="ptitle">Table of Contents</p>

<xsl:for-each select="./part">
<xsl:variable name="partnum"><xsl:number/></xsl:variable>

<p class="phead">
<xsl:value-of select="concat('Part ',$partnum,': ',./title)"/>
</p>

<xsl:for-each select="./chapter">
<xsl:variable name="chapnum"><xsl:number/></xsl:variable>

<p class="psubhead">
<xsl:value-of select="concat($chapnum,' ',./title)"/>
</p>

<ul>
<xsl:for-each select="./sect1">
<xsl:if test="./title!=''">
<li>
<xsl:value-of select="./title"/>
<ul>
<xsl:for-each select="./sect2">
<xsl:if test="./title!=''">
<li><xsl:value-of select="./title"/></li>
</xsl:if>
</xsl:for-each>
</ul>
</li>
</xsl:if>
</xsl:for-each>
</ul>


</xsl:for-each>
</xsl:for-each>

<xsl:for-each select="./part">
<xsl:variable name="partnum"><xsl:number/></xsl:variable>

<!-- all br's translate into forced page breaks (see doc_styles.css) -->
<br/>

<p class="ptitle">
<xsl:value-of select="concat('Part ',$partnum,': ',./title)"/>
</p>

<xsl:for-each select="./chapter">
<xsl:variable name="chapnum"><xsl:number/></xsl:variable>

<!-- all br's translate into forced page breaks (see doc_styles.css) -->
<br/>

<p class="ptitle">
<xsl:value-of select="concat('Chapter ',$chapnum,': ',./title)"/>
</p>

<xsl:for-each select="./sect1">
<xsl:variable name="sectnum"><xsl:number/></xsl:variable>
<xsl:apply-templates mode="print">
<xsl:with-param name="sectn">
<xsl:value-of select="$sectnum"/>
</xsl:with-param>
</xsl:apply-templates>
</xsl:for-each>

</xsl:for-each>
</xsl:for-each>

</xsl:if>

<!-- ******************************************************************* -->
<!-- *********************** STANDARD SCI FOOTER *********************** -->
<!-- ******************************************************************* -->

<xsl:if test="$cont!='printable'">
<div class="banner-margins">
<table border="0" cellspacing="0" cellpadding="0" height="32" width="100%">
<tr>
<td align="left" width="%100">
  <xsl:attribute name="background">
    <xsl:value-of select="concat($swidk, '/doc/images/banner_bottom_fill.jpg')"/>
  </xsl:attribute>
<img width="444" height="32" border="0">
  <xsl:attribute name="src">
    <xsl:value-of select="concat($swidk, '/doc/images/banner_bottom.jpg')"/>
  </xsl:attribute>
</img>
</td>
</tr>
</table>
<center><font size="-2" face="arial, helvetica, sans-serif">Scientific
Computing and Imaging Institute &#149; 50 S. Central Campus Dr. Rm
3490 &#149; Salt Lake City, UT 84112<br />

(801) 585-1867 &#149; fax: (801) 585-6513 &#149; <a href="http://www.utah.edu/disclaimer/disclaimer_home.html">Disclaimer</a></font></center>
</div>
</xsl:if>

<!-- ********************* END STANDARD SCI FOOTER ********************* -->
<!-- ******************************************************************* -->

</body>
</html>
</xsl:template>

</xsl:stylesheet>
