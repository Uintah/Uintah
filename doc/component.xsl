<?xml version="1.0"?>

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:param name="dir"/>

<xsl:template match="/component">
<xsl:processing-instruction name="cocoon-format">type="text/html"</xsl:processing-instruction>

<html>
<head>

<xsl:variable name="swidk">
<xsl:choose>
  <xsl:when test="$dir=4">../../../..</xsl:when>
  <xsl:when test="$dir=3">../../..</xsl:when>
  <xsl:when test="$dir=2">../..</xsl:when>
  <xsl:when test="$dir=1">..</xsl:when>
</xsl:choose>
</xsl:variable>

<title><xsl:value-of select="@category" /> -&#62; <xsl:value-of select="@name" /></title>
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
<center><img SRC="http://www.sci.utah.edu/research/images/research_header_sm.gif" usemap="#head-links" height="46" width="600" border="0" /></center>
<map name="head-links">
	<area shape="rect" coords="491,15,567,32" href="http://www.sci.utah.edu/research/research.html" />
	<area shape="rect" coords="31,9,95,36" href="http://www.sci.utah.edu" />
</map>
<!-- *************************************************************** -->
<!-- *************************************************************** -->

<p class="title"><xsl:value-of select="@category" /> -&#62; <xsl:value-of select="@name" /></p>

<!-- ************************* -->
<!-- ******** OVERVIEW ******* -->
<!-- ************************* -->

<center>Author(s):
<xsl:for-each select="overview/authors/author">
	<xsl:value-of select="." />, 
</xsl:for-each><br />
</center>

<p class="head">Overview</p>
<p class="subhead">Summary</p>
<p class="para">
<xsl:value-of select="overview/summary" />
</p>

<p class="subhead">Description</p>
<xsl:for-each select="overview/description/p">
	<p class="para">
	<xsl:value-of select="." />
	</p>
</xsl:for-each>

<p class="element">For an example of how this component is used, see this .sr file:
<xsl:value-of select="overview/examplesr" />
</p>

<!-- ************************* -->
<!-- ***** IMPLEMENTATION **** -->
<!-- ************************* -->

<xsl:for-each select="implementation">
	<p class="subhead">Implementation</p>
	<p class="element">The following files are required for the build: </p>
	<ul class="element">
	<xsl:for-each select="ccfile">
		<li><b><xsl:value-of select="." /></b></li>
	</xsl:for-each>

	<xsl:for-each select="cfile">
		<li><b><xsl:value-of select="." /></b></li>
	</xsl:for-each>

	<xsl:for-each select="ffile">
		<li><b><xsl:value-of select="." /></b></li>
	</xsl:for-each>
	</ul>
</xsl:for-each>

<!-- ************************* -->
<!-- ********** I/O ********** -->
<!-- ************************* -->

<p class="head">I/O</p>

<!-- ********* INPUTS ******** -->
<!-- ************************* -->

<p class="subhead">Inputs</p>

<xsl:for-each select="io/inputs/file">
	<p class="element">
	Type: <b>File</b><br />
	Datatype: <b><xsl:value-of select="datatype" /></b><br />
	Description:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p><hr width="300" size="1"  />
	</xsl:for-each>
</xsl:for-each><br />

<xsl:for-each select="io/inputs/port">
	<p class="element">
	Name: <b><xsl:value-of select="name" /></b><br />
	Type: <b>Port</b><br />
	Description:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p>
	</xsl:for-each>

	<p class="element">Datatype: <b><xsl:value-of select="datatype" /></b><br />
	The following upstream components are commonly used to send data to this component: 
	<b><xsl:for-each select="componentname">
		<xsl:value-of select="." />, 
	</xsl:for-each></b>
	</p><hr width="300" size="1"  />
</xsl:for-each><br />

<xsl:for-each select="io/inputs/device">
	<p class="element"><b><xsl:value-of select="name" /></b><br />
	Type: <b>Device</b><br />
	Datatype: <b><xsl:value-of select="datatype" /></b><br />
	Description:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p><hr width="300" size="1"  />
	</xsl:for-each>
</xsl:for-each><br />

<!-- ******** OUTPUTS ******** -->
<!-- ************************* -->

<p class="subhead">Outputs</p>

<xsl:for-each select="io/outputs/file">
	<p class="element">
	Type: <b>File</b><br />
	Datatype: <b><xsl:value-of select="datatype" /></b><br />
	Description:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p><hr width="300" size="1"  />
	</xsl:for-each>
</xsl:for-each><br />

<xsl:for-each select="io/outputs/port">
	<p class="element">
	Name: <b><xsl:value-of select="name" /></b><br />
	Type: <b>Port</b><br />
	Description:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p>
	</xsl:for-each>
	<p class="element">Datatype: <b><xsl:value-of select="datatype" /></b><br />
	Components: 
	<b><xsl:for-each select="componentname">
		<xsl:value-of select="." />, 
	</xsl:for-each></b>
	</p><hr width="300" size="1"  />
</xsl:for-each><br />

<xsl:for-each select="io/outputs/device">
	<p class="element"><b><xsl:value-of select="name" /></b><br />
	Type: <b>Device</b><br />
	Datatype: <b><xsl:value-of select="datatype" /></b><br />
	Description:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p><hr width="300" size="1"  />
	</xsl:for-each>
</xsl:for-each><br />

<!-- ************************* -->
<!-- ********** GUI ********** -->
<!-- ************************* -->

<xsl:for-each select="gui">
	<p class="head">GUI</p>

	<center>
	<img border="0">
		<xsl:attribute name="src">
			<xsl:value-of select="img" />
		</xsl:attribute>
	</img></center>
	
	<p class="element"><b>Description :</b></p>

	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p>
	</xsl:for-each>

	<p class="element"><b>Parameters:</b></p>

	<xsl:for-each select="parameter">
		<p class="element">Widget: <b><xsl:value-of select="widget" /></b><br />
		Label: <b><xsl:value-of select="label" /></b><br />
		Description: <b><xsl:value-of select="description" /></b><br />
		</p>
	</xsl:for-each>
</xsl:for-each><br />

<!-- ***************************** -->
<!-- ********** TESTING ********** -->
<!-- ***************************** -->

<p class="head">Testing Plan(s)</p>

<xsl:for-each select="testing/plan">
	<p class="element">Plan:</p>
	<xsl:for-each select="description/p">
		<p class="para">
		<xsl:value-of select="." />
		</p>
	</xsl:for-each>

	<p class="element">Steps:</p>
	<xsl:for-each select="step/p">
		<p class="para">
		<xsl:value-of select="." />
		</p>
	</xsl:for-each><hr width="300" size="1"  />
</xsl:for-each><br />

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
