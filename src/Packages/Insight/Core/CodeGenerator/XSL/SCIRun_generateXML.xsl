<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

<xsl:output method="xml" indent="yes"/>


<!-- ********* GLOBALS ********* -->
<xsl:variable name="sci-name">
  <xsl:value-of select="/filter/filter-sci/@name"/>
</xsl:variable>
<xsl:variable name="itk-name">
  <xsl:value-of select="/filter/filter-itk/@name"/>
</xsl:variable>
<xsl:variable name="package">
  <xsl:value-of select="/filter/filter-sci/package"/>
</xsl:variable>
<xsl:variable name="category">
  <xsl:value-of select="/filter/filter-sci/category"/>
</xsl:variable>
<xsl:variable name="base">
  <xsl:value-of select="/filter/filter-itk/templated/base"/>
</xsl:variable>



<!-- SKIP INFORMATION FROM SCIRUN and GUI XML FILES -->
<xsl:template match="filter-sci">
</xsl:template>

<xsl:template match="filter-gui">
</xsl:template>

<xsl:template match="filter-sci">
</xsl:template>

<xsl:template match="define-gui">
</xsl:template>


<!-- NEW ITK FILTER -->
<xsl:template match="filter-itk">
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


<!-- INPUT PORTS -->
<xsl:template match="inputs"><xsl:text>
    </xsl:text><xsl:element name="inputs">
<xsl:attribute name="lastportdynamic">no</xsl:attribute>
<xsl:apply-templates select="input"/><xsl:text>
    </xsl:text></xsl:element><xsl:text>
  </xsl:text>
</xsl:template>

<xsl:template match="input">
<xsl:variable name="type-name"><xsl:value-of select="value"/></xsl:variable>

<xsl:text>  
      </xsl:text>
 <xsl:element name="port"><xsl:text>
        </xsl:text><xsl:element name="name"><xsl:value-of select="name"/></xsl:element><xsl:text>
        </xsl:text><xsl:element name="datatype"><xsl:value-of select="$package"/><xsl:text>::</xsl:text><xsl:value-of select="/filter/filter-sci/instantiations/instance/type[@name=$type-name]/datatype"/></xsl:element><xsl:text>
      </xsl:text>
    </xsl:element>
</xsl:template>


<!-- OUTPUT PORTS -->
<xsl:template match="outputs">
<xsl:text>  </xsl:text><xsl:element name="outputs">
<xsl:apply-templates select="output"/><xsl:text>
    </xsl:text></xsl:element>
</xsl:template>

<xsl:template match="output">
<xsl:variable name="type-name"><xsl:value-of select="value"/></xsl:variable>
<xsl:text>  
      </xsl:text>
 <xsl:element name="port"><xsl:text>
        </xsl:text><xsl:element name="name"><xsl:value-of select="name"/></xsl:element><xsl:text>
        </xsl:text><xsl:element name="datatype"><xsl:value-of select="$package"/><xsl:text>::</xsl:text><xsl:value-of select="/filter/filter-sci/instantiations/instance/type[@name=$type-name]/datatype"/></xsl:element><xsl:text>
      </xsl:text>
    </xsl:element>
</xsl:template>




<!-- ******************* HELPER FUNCTIONS ********************  -->
<!--header info -->
<xsl:template name="header">
<xsl:text disable-output-escaping="yes">&lt;!DOCTYPE component SYSTEM &quot;../../../../../doc/Utilities/XML/component.dtd&quot;&gt;
&lt;?xml-stylesheet href=&quot;../../../../../doc/Utilities/XML/package-component.xsl&quot;type=&quot;text/xsl&quot;?&gt;
&lt;?cocoon-process type=&quot;xslt&quot;?&gt;
</xsl:text>
</xsl:template>

</xsl:stylesheet>