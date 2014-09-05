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
<xsl:param name="treetop"/>
<xsl:param name="cont"/>

<xsl:include href="top_banner.xsl"/>
<xsl:include href="bottom_banner.xsl"/>

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

<html>

<head>
<title><xsl:value-of select="./bookinfo/title" /></title>
<link rel="stylesheet" type="text/css">
<xsl:attribute name="href">
  <xsl:value-of select="concat($treetop,'doc/Utilities/HTML/doc_styles.css')" />
</xsl:attribute>
</link>
</head>

<body>

<!-- *************************************************************** -->
<!-- *************** STANDARD SCI RESEARCH HEADER ****************** -->
<!-- *************************************************************** -->

<xsl:if test="$cont!='printable'">
<xsl:call-template name="top_banner"/>
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
<img src="../../Utilities/Figures/SCI_logo.jpg" vspace="50"/>
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
<xsl:call-template name="bottom_banner"/>
</xsl:if>

<!-- ********************* END STANDARD SCI FOOTER ********************* -->
<!-- ******************************************************************* -->

</body>
</html>
</xsl:template>

</xsl:stylesheet>
