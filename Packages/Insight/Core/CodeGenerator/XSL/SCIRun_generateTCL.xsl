<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

<xsl:output method="text" indent="no"/>

<!-- ============ GLOBAL VARIABLES ============ -->
<xsl:variable name="has_gui"><xsl:value-of select="/filter/filter-gui"/></xsl:variable>

<!-- ============== /FILTER =====================-->
<xsl:template match="/filter">
<!-- Insert copyright -->
<xsl:text>#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#
</xsl:text>
<xsl:text> itcl_class Insight_Filters_</xsl:text>
<!-- Constructor -->
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text> {
    inherit Module
    constructor {config} {
         set name </xsl:text>
<xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text>
</xsl:text>
<!-- Create globals corresponding to parameters -->
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
</xsl:text>
<!-- Create set_defaults function -->
<xsl:text>
    method set_defaults {} {
</xsl:text>
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
</xsl:text>
<!-- Create ui function -->
<xsl:choose>
<xsl:when test="$has_gui = ''">
</xsl:when>
<xsl:otherwise>
<xsl:text>
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
    <xsl:with-param name="from" select="min"/>
    <xsl:with-param name="to" select="max"/>
  </xsl:call-template>
</xsl:when>
<xsl:when test="$gui = 'checkbutton'">
  <xsl:call-template name="create_checkbutton"/>
</xsl:when>
<xsl:when test="$gui = 'radiobutton'">
  <xsl:call-template name="create_radiobutton">
    <xsl:with-param name="values" select="values"/>
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
</xsl:text>
</xsl:otherwise>
</xsl:choose>
<xsl:text>
}

</xsl:text>

</xsl:template>

<!--  ================= WIDGET FUNCTIONS ================= -->
<xsl:template name="execute_and_close_buttons">
<xsl:text disable-output-escaping="yes">        
        button $w.execute -text &quot;Execute&quot; -command &quot;$this-c needexecute&quot;
        button $w.close -text &quot;Close&quot; -command &quot;destroy $w&quot;
        pack $w.execute $w.close -side left
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
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> \
        -textvariable $this-</xsl:text><xsl:value-of select="name"/><xsl:text>
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

<!-- CREATE_SCROLLBAR -->
<xsl:template name="create_scrollbar">
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


<!-- CREATE_CHECKBUTTON -->
<xsl:template name="create_checkbutton">
<xsl:variable name="widget">checkbutton</xsl:variable>
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="@name"/>
</xsl:variable>
<xsl:variable name="default"><xsl:value-of select="default"/></xsl:variable>
<xsl:text>
        frame </xsl:text><xsl:value-of select="$path"/><xsl:text>
        </xsl:text>
<xsl:value-of select="$widget"/>
<xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/> \
           -text &quot;<xsl:value-of select="@name"/>&quot; \
<xsl:text>           -variable $this-</xsl:text><xsl:value-of select="@name"/>
        pack <xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> -side left
</xsl:text>
<xsl:text>        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>

</xsl:text>
</xsl:template>


<!-- CREATE_RADIOBUTTON -->
<xsl:template name="create_radiobutton">
  <xsl:param name="values"/>
<xsl:variable name="name"><xsl:value-of select="@name"/></xsl:variable>
<xsl:variable name="widget">radiobutton</xsl:variable>
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="$name"/>
</xsl:variable>
<xsl:text>
        frame </xsl:text><xsl:value-of select="$path"/>
        label <xsl:value-of select="$path"/>.label -text <xsl:value-of select="$name"/>
        pack <xsl:value-of select="$path"/>.label
<xsl:for-each select="values/val">
<xsl:text>
        </xsl:text>
<xsl:value-of select="$widget"/><xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="position()"/> \
           -text &quot;<xsl:value-of select="."/>&quot; \
<xsl:text>           -variable $this-</xsl:text><xsl:value-of select="$name"/> \
           -value <xsl:value-of select="."/>
        pack <xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="position()"/><xsl:text> -side left
</xsl:text>
</xsl:for-each>
<xsl:text>        pack </xsl:text> <xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>	     
</xsl:template>
</xsl:stylesheet>
