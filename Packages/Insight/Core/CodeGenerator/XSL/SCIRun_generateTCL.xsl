<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

<xsl:output method="text" indent="no"/>
<xsl:variable name="has_gui"><xsl:value-of select="/filter/filter-gui"/></xsl:variable>

<!-- FILTER -->
<xsl:template match="filter">
<xsl:text> itcl_class Insight_Filters_</xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text> {
    inherit Module
    constructor {config} {
         set name </xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text>
</xsl:text>
<!-- HERE -->
<xsl:choose>
<xsl:when test="$has_gui = ''">
<xsl:for-each select="/filter/filter-itk/parameters/param">
<xsl:text>
         global $this-</xsl:text> 
<xsl:value-of select="name"/>  
</xsl:for-each>
</xsl:when>
<xsl:otherwise>
<xsl:for-each select="/filter/filter-gui/parameters/param">
<xsl:text>
         global $this-</xsl:text> 
<xsl:value-of select="@name"/>  
</xsl:for-each>
</xsl:otherwise>
</xsl:choose><xsl:text>

         set_defaults
    }

    method set_defaults {} {
</xsl:text>
<!-- HERE -->
<xsl:choose>
<xsl:when test="$has_gui = ''">
<xsl:for-each select="/filter/filter-itk/parameters/param">
<xsl:text>
         set $this-</xsl:text> 
<xsl:value-of select="name"/><xsl:text> 0</xsl:text> 
</xsl:for-each>
</xsl:when>
<xsl:otherwise>
<xsl:for-each select="/filter/filter-gui/parameters/param">
<xsl:text>
         set $this-</xsl:text> 
<xsl:value-of select="@name"/><xsl:text> </xsl:text>  
<xsl:value-of select="default"/>
</xsl:for-each>
</xsl:otherwise>
</xsl:choose>
<xsl:text>    
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }
        toplevel $w
</xsl:text>
<!-- Parameters -->
<!-- HERE -->
<xsl:choose>
<xsl:when test="$has_gui = ''">
<xsl:for-each select="/filter/filter-itk/parameters/param">
<xsl:variable name="gui"><xsl:value-of select="gui"/></xsl:variable>
  <xsl:call-template name="create_text_entry">
  </xsl:call-template>
</xsl:for-each>
</xsl:when>
<xsl:otherwise>
<xsl:for-each select="/filter/filter-gui/parameters/param">
<xsl:variable name="gui"><xsl:value-of select="gui"/></xsl:variable>
<xsl:choose>
<xsl:when test="$gui = 'text-entry'">
  <xsl:call-template name="create_text_entry">
  </xsl:call-template>
</xsl:when>
<xsl:when test="$gui = 'scrollbar'">
  <xsl:call-template name="create_scrollbar">
    <xsl:with-param name="gui" select="$gui"/>
    <xsl:with-param name="from" select="min"/>
    <xsl:with-param name="to" select="max"/>
  </xsl:call-template>
</xsl:when>
<xsl:otherwise>
  <!-- not defined, default to text-entry -->
  <xsl:call-template name="create_text_entry">
  </xsl:call-template>
</xsl:otherwise>
</xsl:choose>

</xsl:for-each>
</xsl:otherwise>
</xsl:choose>
<xsl:call-template name="execute_and_close_buttons"/>
<xsl:text>
    }
}

</xsl:text>

</xsl:template>


<!-- HELPER FUNCTIONS -->
<xsl:template name="execute_and_close_buttons">
<xsl:text disable-output-escaping="yes">        
        button $w.execute -text &quot;Execute&quot; -command &quot;$this-c needexecute&quot;
        button $w.close -text &quot;Close&quot; -command &quot;destroy $w&quot;
        pack $w.execute $w.close -side top
</xsl:text>
</xsl:template>

<!-- CREATE_TEXT_ENTRY -->
<xsl:template name="create_text_entry">
<xsl:variable name="widget">entry</xsl:variable>
<xsl:choose>
<xsl:when test="$has_gui = ''">
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="name"/>
</xsl:variable>
<xsl:text>
        frame </xsl:text><xsl:value-of select="$path"/><xsl:text>
        label </xsl:text><xsl:value-of select="$path"/><xsl:text disable-output-escaping="yes">.label -text &quot;</xsl:text><xsl:value-of select="name"/><xsl:text disable-output-escaping="yes">&quot;
        </xsl:text>
<xsl:value-of select="$widget"/>
<xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/>
<xsl:text> -textvariable $this-</xsl:text><xsl:value-of select="name"/><xsl:text>
        pack </xsl:text><xsl:value-of select="$path"/><xsl:text>.label </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> -side left
</xsl:text>
<xsl:text>        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>
</xsl:when>
<xsl:otherwise>
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="@name"/>
</xsl:variable>
<xsl:text>
        frame </xsl:text><xsl:value-of select="$path"/><xsl:text>
        label </xsl:text><xsl:value-of select="$path"/><xsl:text disable-output-escaping="yes">.label -text &quot;</xsl:text><xsl:value-of select="@name"/><xsl:text disable-output-escaping="yes">&quot;
        </xsl:text>
<xsl:value-of select="$widget"/>
<xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/>
<xsl:text> -textvariable $this-</xsl:text><xsl:value-of select="@name"/><xsl:text>
        pack </xsl:text><xsl:value-of select="$path"/><xsl:text>.label </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> -side left
</xsl:text>
<xsl:text>        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>
</xsl:otherwise>
</xsl:choose>
</xsl:template>

<xsl:template name="create_scrollbar">
  <xsl:param name="gui"/>
  <xsl:param name="from"/>
  <xsl:param name="to"/>

<xsl:variable name="widget">scale</xsl:variable>
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="@name"/>
</xsl:variable>
<xsl:text>
        frame </xsl:text><xsl:value-of select="$path"/><xsl:text>
        </xsl:text>
<xsl:value-of select="$widget"/>
<xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/>
<xsl:text> -label </xsl:text> &quot;<xsl:value-of select="@name"/>&quot; \
<xsl:text>           -variable $this-</xsl:text><xsl:value-of select="@name"/> \
           -from <xsl:value-of select="$from"/> -to <xsl:value-of select="$to"/> -orient horizontal<xsl:text>
        pack </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> -side left
</xsl:text>
<xsl:text>        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>	     
</xsl:template>

</xsl:stylesheet>