<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

<xsl:output method="xml" indent="yes"/>

<!-- ======================================= -->
<!-- =============== GLOBALS =============== -->
<!-- ======================================= -->
<!-- Variable sci-name refers to the name attribute of
     the sci xml file.  This is the name that the module
     will be referred to in SCIRun.
-->
<xsl:variable name="sci-name">
  <xsl:value-of select="/filter/filter-sci/@name"/>
</xsl:variable>

<!-- Variable itk-name refers to the name attribute of
     the itk xml file.  This name includes the namespace
     of the itk filter.
-->
<xsl:variable name="itk-name">
  <xsl:value-of select="/filter/filter-itk/@name"/>
</xsl:variable>

<!-- Variable package indicates the SCIRun package
     this module/filter will belong to.
-->
<xsl:variable name="package">
  <xsl:value-of select="/filter/filter-sci/package"/>
</xsl:variable>

<!-- Variable category referst to the SCIRun category
     this module/filter will belong to.
-->
<xsl:variable name="category">
  <xsl:value-of select="/filter/filter-sci/category"/>
</xsl:variable>


<!-- ======================================== -->
<!-- ================ FILTER ================ -->
<!-- ======================================== -->
<!-- Tag filter-itk is the root of the itk xml file 
     so just call that one and skip the others.-->
<xsl:template match="/filter">
  <xsl:apply-templates select="filter-itk"/>
</xsl:template>


<!-- ======================================== -->
<!-- ============== FILTER-ITK ============== -->
<!-- ======================================== -->
<xsl:template match="filter-itk">
<!-- Add SCIRun copyright -->
<xsl:call-template name="copyright"/>

<!-- Add standard SCIRun xml header information -->
<xsl:call-template name="header"/>

<xsl:element name="component">
<xsl:attribute name="name"><xsl:value-of select="$sci-name"/></xsl:attribute>
<xsl:attribute name="category"><xsl:value-of select="$category"/></xsl:attribute><xsl:text>
  </xsl:text><xsl:element name="overview"><xsl:text>
    </xsl:text><xsl:element name="authors"><xsl:text>
    </xsl:text></xsl:element><xsl:text>
    </xsl:text><xsl:element name="summary"><xsl:text>    </xsl:text>
    <xsl:value-of select="description"/>
<xsl:text>  </xsl:text></xsl:element><xsl:text>
  </xsl:text></xsl:element><xsl:text>
  </xsl:text><xsl:element name="implementation"><xsl:text>
  </xsl:text></xsl:element><xsl:text>
  </xsl:text><xsl:element name="io">
  <xsl:apply-templates select="inputs"/>
  <xsl:apply-templates select="outputs"/><xsl:text>
  </xsl:text></xsl:element>
 </xsl:element>
</xsl:template>




<!-- =================================== -->
<!-- ============= INPUTS ============== -->
<!-- =================================== -->
<xsl:template match="inputs"><xsl:text>
    </xsl:text><xsl:element name="inputs">
<xsl:attribute name="lastportdynamic">no</xsl:attribute>
<!-- Each input port has a name and a datatype.  The name 
     corresponds to the name given in the itk xml file and
     for now the datatype is always Insight::ITKDatatype.
-->
<xsl:for-each select="input">
<xsl:variable name="type-name"><xsl:value-of select="value"/></xsl:variable>
<xsl:text>  
      </xsl:text>
 <xsl:element name="port"><xsl:text>
        </xsl:text><xsl:element name="name"><xsl:value-of select="@name"/></xsl:element><xsl:text>
        </xsl:text><xsl:element name="datatype"><!-- hard coded datatype -->Insight::ITKDatatype</xsl:element><xsl:text>
      </xsl:text>
    </xsl:element>
</xsl:for-each><xsl:text>
    </xsl:text></xsl:element><xsl:text>
  </xsl:text>
</xsl:template>



<!-- ========================================== -->
<!-- ================ OUTPUTS ================= -->
<!-- ========================================== -->
<xsl:template match="outputs">
<xsl:text>  </xsl:text><xsl:element name="outputs">
<!-- Each input port has a name and a datatype.  The name 
     corresponds to the name given in the itk xml file and
     for now the datatype is always Insight::ITKDatatype.
-->
<xsl:for-each select="output">
<xsl:variable name="type-name"><xsl:value-of select="value"/></xsl:variable>
<xsl:text>  
      </xsl:text>
 <xsl:element name="port"><xsl:text>
        </xsl:text><xsl:element name="name"><xsl:value-of select="@name"/></xsl:element><xsl:text>
        </xsl:text><xsl:element name="datatype"><!-- hard coded datatype -->Insight::ITKDatatype</xsl:element><xsl:text>
      </xsl:text>
    </xsl:element>

</xsl:for-each><xsl:text>
    </xsl:text></xsl:element>
</xsl:template>


<!-- ========================================= -->
<!-- ============ HEADER INFORMATION ========= -->
<!-- ========================================= -->
<xsl:template name="header">
<xsl:text disable-output-escaping="yes">&lt;!DOCTYPE component SYSTEM &quot;../../../../../doc/Utilities/XML/component.dtd&quot;&gt;
&lt;?xml-stylesheet href=&quot;../../../../../doc/Utilities/XML/component.xsl&quot;type=&quot;text/xsl&quot;?&gt;
&lt;?cocoon-process type=&quot;xslt&quot;?&gt;
</xsl:text>
</xsl:template>


<!-- ======================================== -->
<!-- ============== COPYRIGHT =============== -->
<!-- ======================================== -->
<xsl:template name="copyright">
<xsl:text disable-output-escaping="yes">&lt;!--
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
--&gt;

&lt;!-- This is an automatically generated file for the
     </xsl:text><xsl:value-of select="$itk-name"/><xsl:text disable-output-escaping="yes">
--&gt;

</xsl:text>
</xsl:template>

</xsl:stylesheet>
