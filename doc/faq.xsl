<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>
<xsl:param name="cont"/>

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

</xsl:if>

<!-- *************************************************************** -->
<!-- *************************************************************** -->

<xsl:variable name="source">
  <xsl:value-of select="/faqbook/bookinfo/title"/>
</xsl:variable>

<xsl:if test="$cont=0">

<!-- ************** Table of Contents ****************************** -->

<table border="0"><tr><td width="50">
  PREV
</td><td>
<a>
  <xsl:attribute name="href">
    <xsl:value-of
      select="concat($source,'?dir=2&amp;cont=1')"/>
  </xsl:attribute>
  NEXT
</a>
</td></tr></table>

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
        <xsl:value-of select="concat($source,'?dir=2&amp;cont=',$faqnum)"/>
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
      <xsl:when test="$faqnum&gt;0">
      <a>
        <xsl:attribute name="href">
          <xsl:value-of
            select="concat($source,'?dir=2&amp;cont=',$prev)"/>
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
          <xsl:value-of
            select="concat($source,'?dir=2&amp;cont=',$next)"/>
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
