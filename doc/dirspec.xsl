<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>

<xsl:template match="/dir">
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
<title>SCIRun/<xsl:value-of select="@name" /></title>
<link rel="stylesheet" type="text/css">
<xsl:attribute name="href">
<xsl:value-of select="concat($swidk,'/doc/doc_styles.css')" />
</xsl:attribute>
</link>
<!-- *************************************************************** -->
<!-- ******************* BEGIN HEIRMENU SCRIPT ********************* -->
<!-- ****************** (Place in document HEAD) ******************* -->
<!-- *************************************************************** -->
<SCRIPT LANGUAGE="JavaScript" TYPE="text/javascript">
       <!--
 
       if(window.event + "" == "undefined") event = null;
       function HM_f_PopUp(){return false};
       function HM_f_PopDown(){return false};
       popUp = HM_f_PopUp;
       popDown = HM_f_PopDown;
 
       //-->
       </SCRIPT>

       <SCRIPT LANGUAGE="JavaScript1.2" TYPE="text/javascript">
       <!--
 
       HM_PG_MenuWidth = 150;
       HM_PG_FontFamily = "Arial, helvetica, sans-serif";
       HM_PG_FontSize = 10;
       HM_PG_FontBold = 1;
       HM_PG_FontItalic = 0;
       HM_PG_FontColor = "#000000";
       HM_PG_FontColorOver = "#FFFFFF";
       HM_PG_BGColor = "#CCCCCC";
       HM_PG_BGColorOver = "#999999";
       HM_PG_ItemPadding = 3; 
 
       HM_PG_BorderWidth = 2;
       HM_PG_BorderColor = "#000000";
       HM_PG_BorderStyle = "solid";
       HM_PG_SeparatorSize = 2;
       HM_PG_SeparatorColor = "#000000";
       HM_PG_ImageSrc = "tri.gif";
       HM_PG_ImageSrcLeft = "triL.gif";
 
       HM_PG_ImageSize = 12;
       HM_PG_ImageHorizSpace = 0;
       HM_PG_ImageVertSpace = 2;
                                                                                      HM_PG_KeepHilite = true;
       HM_PG_ClickStart = 0;
       HM_PG_ClickKill = false;
       HM_PG_ChildOverlap = 20;
       HM_PG_ChildOffset = 10;
       HM_PG_ChildPerCentOver = null;
       HM_PG_TopSecondsVisible = .3;
       HM_PG_StatusDisplayBuild =1;
       HM_PG_StatusDisplayLink = 1;
       HM_PG_UponDisplay = null;
       HM_PG_UponHide = null;
       HM_PG_RightToLeft = false;
 
       //HM_PG_CreateTopOnly = 1;
       HM_PG_ShowLinkCursor = 1;
 
       //HM_a_TreesToBuild = [1,2];
 
       //-->
       </SCRIPT>                                                   
         
<SCRIPT LANGUAGE="JavaScript1.2" TYPE='text/javascript'>
<xsl:attribute name="src">
<xsl:value-of select="concat($swidk,'/doc/HM_Loader.js')" />
</xsl:attribute>
</SCRIPT>
<!-- *************************************************************** -->
<!-- ******************** END HEIRMENU SCRIPT ********************** -->
<!-- *************************************************************** -->       
</head>
<body>

<!-- *************************************************************** -->
<!-- *************** STANDARD SCI RESEARCH HEADER ****************** -->
<!-- *************************************************************** -->

<center>
<img usemap="#head-links" height="71" width="600" border="0">
<xsl:attribute name="src">
<xsl:value-of select="concat($swidk,'/doc/images/research_menuheader.gif')" />
</xsl:attribute>
</img>
</center>
<map name="head-links">
        <area shape="rect" coords="491,15,567,32" href="http://www.sci.utah.edu/research/research.html"/>
        <area shape="rect" coords="31,9,95,36" href="http://www.sci.utah.edu"/>
 
        <area coords="0,45,150,70" shape="rect" onMouseOver="HM_f_PopUp('elMenu1',event)" onMouseOut="HM_f_PopDown('elMenu1')">
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/InstallGuide/installguide.html')" />
        </xsl:attribute>
        </area>

 
        <area coords="150,45,300,70" shape="rect" onMouseOver="HM_f_PopUp('elMenu2',event)" onMouseOut="HM_f_PopDown('elMenu2')"> 
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/UserGuide/userguide.html')" />
        </xsl:attribute>
        </area>

        <area coords="300,45,450,70" shape="rect" onMouseOver="HM_f_PopUp('elMenu3',event)" onMouseOut="HM_f_PopDown('elMenu3')">
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/DeveloperGuide/devguide.html')" />
        </xsl:attribute>
        </area>
 
        <area coords="450,45,600,70" shape="rect" onMouseOver="HM_f_PopUp('elMenu4',event)" onMouseOut="HM_f_PopDown('elMenu4')">  
        <xsl:attribute name="href">
        <xsl:value-of select="concat($swidk,'/doc/ReferenceGuide/refguide.html')" />
        </xsl:attribute>
        </area>
</map> 

<!-- *************************************************************** -->
<!-- *************************************************************** -->


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
