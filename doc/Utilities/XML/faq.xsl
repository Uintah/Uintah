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

<xsl:template match="para" mode="question">
  <xsl:param name="num"/>
  <p class="question">
    <b><xsl:apply-templates/></b>
  </p>
</xsl:template>

<xsl:template match="pre" mode="question">
  <div class="box"><br/>
    <pre class="example">
      <font color="blue"><b>
        <xsl:apply-templates/>
      </b></font>
    </pre>
  </div>
</xsl:template>

<xsl:template match="para" mode="answer">
  <p class="firstpara"><xsl:apply-templates/></p>
</xsl:template>

<xsl:template match="pre" mode="answer">
  <div class="box"><br/>
    <pre class="example">
      <font color="blue"><xsl:apply-templates/></font>
    </pre>
  </div>
</xsl:template>

<xsl:template match="para" mode="print">
  <xsl:param name="num"/>
  <p class="pfirstpara">
    <xsl:apply-templates mode="print"/>
  </p>
</xsl:template>

<xsl:template match="pre" mode="print">
  <div class="box">
    <pre class="pexample">

      <font color="blue"><b>
        <xsl:apply-templates mode="print"/>
      </b></font>
    </pre>
  </div>
</xsl:template>

<xsl:template match="a" mode="print">
  <xsl:apply-templates mode="print"/>
</xsl:template>

<xsl:template match="@*|node()" mode="print">
  <xsl:copy>
    <xsl:apply-templates select="@*" mode="print"/>
    <xsl:apply-templates mode="print"/>
  </xsl:copy>
</xsl:template>


<xsl:template match="@*|node()">
  <xsl:copy>
    <xsl:apply-templates select="@*"/>
    <xsl:apply-templates />
  </xsl:copy>
</xsl:template>

<xsl:template match="/faqbook">
<xsl:processing-instruction name="cocoon-format">type="text/html"</xsl:processing-instruction>
<html>

<head>
<title><xsl:value-of select="./bookinfo/title" /></title>
<link rel="stylesheet" type="text/css">
<xsl:attribute name="href">
  <xsl:value-of select="concat($treetop,'doc/Utilities/HTML/faq.css')" />
</xsl:attribute>
</link>

</head>
<body>

<!-- *************************************************************** -->
<!-- *************** STANDARD SCI RESEARCH HEADER ****************** -->
<!-- *************************************************************** -->

<xsl:if test="$cont!='print'">
<xsl:call-template name="top_banner"/>
</xsl:if>

<!-- *************************************************************** -->
<!-- *************************************************************** -->

<xsl:variable name="source">
  <xsl:value-of select="/faqbook/bookinfo/title"/>
</xsl:variable>

<xsl:if test="$cont=0">

<!-- ************** Table of Contents ****************************** -->

<!-- <table border="0"><tr><td width="50">
  PREV
</td><td>
<a>
  <xsl:attribute name="href">
    <xsl:value-of
      select="concat($source,'?dir=2&amp;cont=1')"/>
  </xsl:attribute>
  NEXT
</a>
</td></tr></table> -->

<p class="title">
  <xsl:value-of select="./title"/>
</p>

<p class="subtitle">
  <xsl:value-of select="./subtitle"/>
</p>

<hr size="1"/>

<xsl:for-each select="./description">
  <xsl:apply-templates/>
</xsl:for-each>

<hr size="1"/>

<xsl:for-each select="./faq">
  <xsl:variable name="faqnum"><xsl:number/></xsl:variable>

  <p class="head">
    <xsl:value-of select="$faqnum"/>
    <xsl:value-of select="concat(' ',' ')"/>
    <a>
      <xsl:attribute name="href">
        <xsl:value-of select="concat('faq',$faqnum,'.html')"/>
      </xsl:attribute>
      <xsl:value-of select="./title"/>
    </a>
  </p>

  <p class="firstpara">
    <xsl:apply-templates select="description"/>
  </p>

</xsl:for-each>

</xsl:if>

<xsl:if test="$cont>0">
  
<!-- *********** FAQ's ************ -->

<xsl:for-each select="./faq">
  <xsl:variable name="faqnum"><xsl:number/></xsl:variable>
  <xsl:variable name="prev">
    <xsl:value-of select="$faqnum - 1"/>
  </xsl:variable>
  <xsl:variable name="next">
    <xsl:value-of select="$faqnum + 1"/>
  </xsl:variable>

  <xsl:if test="$faqnum=$cont">
    <table border="0"><tr><td width="50">
    <xsl:choose>
      <xsl:when test="$faqnum&gt;1">
      <a>
        <xsl:attribute name="href">
          <xsl:value-of select="concat('faq',$faqnum - 1,'.html')"/>
        </xsl:attribute>
        PREV
        <xsl:value-of select="concat(' ',' ')"/>
      </a>
      </xsl:when>
      <xsl:otherwise>
        PREV
      </xsl:otherwise>
    </xsl:choose>
    </td>

    <td width="50">
    <xsl:choose>
    <xsl:when test="$faqnum&lt;last()">
      <a>
        <xsl:attribute name="href">
          <xsl:value-of select="concat('faq',$faqnum + 1,'.html')"/>
        </xsl:attribute>
        NEXT
      </a>
    </xsl:when>
    <xsl:otherwise>
      NEXT
    </xsl:otherwise>
    </xsl:choose>
    </td></tr></table>

    <p class="title">
      <xsl:value-of select="./title" />
    </p>

    <p class="firstpara">
      <xsl:apply-templates select="./description"/>
    </p>

    <hr size="1"/>

    <ol>
    <xsl:for-each select="./entry">
      <xsl:variable name="num"><xsl:number/></xsl:variable>
      <xsl:for-each select="./question">
        <xsl:for-each select="./para">
          <xsl:variable name="num2"><xsl:number/></xsl:variable>
          <xsl:if test="$num2=1">
            <li><p class="firstpara">
              <a><xsl:attribute name="href">
                <xsl:value-of select="concat('#',$num)"/></xsl:attribute>
                <xsl:apply-templates/>
              </a>
            </p></li>
          </xsl:if>
        </xsl:for-each>
      </xsl:for-each>
    </xsl:for-each>
    </ol>

    <hr size="1" />

    <xsl:for-each select="./entry">
      <xsl:variable name="qnum"><xsl:number/></xsl:variable>

      <!-- Question -->
      <xsl:for-each select="./question">
        <a>
          <xsl:attribute name="name">
            <xsl:value-of select="$qnum"/>
          </xsl:attribute>
          <span class="dropcap">Q: </span>
        </a>
        <xsl:apply-templates mode="question"/>
      </xsl:for-each>

      <!-- Answer -->
      <xsl:for-each select="./answer">
        <span class="dropcap">A: </span>
        <xsl:apply-templates mode="answer"/>
      </xsl:for-each>
      <hr size="1" />
    </xsl:for-each>

  </xsl:if>

</xsl:for-each>

</xsl:if>

<xsl:if test="$cont='print'">

<!-- ************** Print all ************** -->

<p class="title">
  <xsl:value-of select="./title"/>
</p>

<p class="subtitle">
  Version 
  <xsl:value-of select="concat(' ',' ')"/>
  <xsl:value-of select="./bookinfo/edition"/>
</p>

<center>
  <img src="http://www.sci.utah.edu/sci_images/SCI_logo.jpg" vspace="50"/>
</center>

<p class="subtitle">
  <xsl:value-of select="./subtitle"/>
</p>

<hr size="1"/>

<xsl:for-each select="./description">
  <xsl:apply-templates mode="print"/>
</xsl:for-each>

<hr size="1"/>

<p class="fineprint">
  Copyright (c)
  <xsl:value-of select="./bookinfo/copyright/year"/>
  <xsl:value-of select="concat(' ',' ')"/>
  <xsl:value-of select="./bookinfo/copyright/holder"/>
</p>

<br/>

<p class="ptitle">Table of Contents</p>

<xsl:for-each select="./faq">
  <xsl:variable name="faqnum"><xsl:number/></xsl:variable>
  <p class="phead">
    <xsl:value-of select="$faqnum"/>
    <xsl:value-of select="concat(' ',' ')"/>
    <xsl:value-of select="./title"/>
  </p>

  <p class="pfirstpara">
    <xsl:apply-templates select="description" mode="print"/>
  </p>

</xsl:for-each>

<xsl:for-each select="./faq">
  <xsl:variable name="faqnum"><xsl:number/></xsl:variable>
  <br/>

  <p class="ptitle">
    Chapter 
    <xsl:value-of select="concat(' ',' ')"/>
    <xsl:value-of select="$faqnum"/>:
    <xsl:value-of select="concat(' ',' ')"/>
    <xsl:value-of select="./title" />
  </p>

  <p class="pfirstpara">
    <xsl:value-of select="./description/para" />
  </p>

  <xsl:for-each select="./entry">
    <xsl:variable name="qnum"><xsl:number/></xsl:variable>

    <!-- Question -->
    <xsl:for-each select="./question">
      <a>
        <xsl:attribute name="name">
          <xsl:value-of select="$qnum"/>
        </xsl:attribute>
        <span class="pdropcap">Q: </span>
      </a>
      <xsl:apply-templates mode="print"/>
    </xsl:for-each>

    <!-- Answer -->
    <xsl:for-each select="./answer">
      <span class="pdropcap">A: </span>
      <xsl:apply-templates mode="print"/>
    </xsl:for-each>
    <hr size="1" />
  </xsl:for-each>
</xsl:for-each>

</xsl:if>

<!-- ******************************************************************* -->
<!-- *********************** STANDARD SCI FOOTER *********************** -->
<!-- ******************************************************************* -->

<xsl:if test="$cont!='print'">
<xsl:call-template name="bottom_banner"/>
</xsl:if>

<!-- ********************* END STANDARD SCI FOOTER ********************* -->
<!-- ******************************************************************* -->

</body>
</html>
</xsl:template>

</xsl:stylesheet>
