<?xml version="1.0"?> 
<xsl:stylesheet 
  xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

  <xsl:output method="text" indent="no"/>


<!-- ========================================== -->
<!-- ============ GLOBAL VARIABLES ============ -->
<!-- ========================================== -->
<!-- Variable gui_specifed indicates whether a gui xml
     file was specified in the sci xml file.  If no gui
     was specified, all parameters will be represented by
     a text-entry and the UI button will still appear on 
     the module. 
-->
<xsl:variable name="gui_specified">
	<xsl:value-of select="/filter/filter-gui"/>
</xsl:variable>

<!-- Variable no_params indicates when a filter has no
     parameters.  In this case, no UI button should be
     generated. 
-->
<xsl:variable name="no_params">
	<xsl:value-of select="/filter/filter-itk/parameters/param"/>
</xsl:variable>	      



<!-- =========================================== -->
<!-- ================ FILTER =================== -->
<!-- =========================================== -->
<xsl:template match="/filter">
<!-- Insert SCIRun copyright and auto-generated message-->
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

#############################################################
#  This is an automatically generated file for the 
#  </xsl:text><xsl:value-of select="/filter/filter-itk/@name"/>
#############################################################

<!-- define the itcl class -->
 itcl_class Insight_Filters_<xsl:value-of select="/filter/filter-sci/@name"/> {
    inherit Module
    constructor {config} {
         set name <xsl:value-of select="/filter/filter-sci/@name"/>
<xsl:text>
</xsl:text>
<!-- Create globals corresponding to parameters which will be used
     on the C++ side
 -->
<xsl:choose>
<xsl:when test="$gui_specified = ''">
<xsl:for-each select="/filter/filter-itk/parameters/param">
         global $this-<xsl:value-of select="name"/>  
</xsl:for-each>
</xsl:when>
<xsl:otherwise>
<xsl:for-each select="/filter/filter-gui/parameters/param">
<xsl:text>
         global $this-</xsl:text> 
<xsl:value-of select="@name"/>  
</xsl:for-each>
</xsl:otherwise>
</xsl:choose>

         set_defaults
    }
<!-- Create set_defaults method. If the gui isn't
     specified, run for loop over itk xml parameters
     and initialize everything to 0
-->
    method set_defaults {} {
<xsl:choose>
<xsl:when test="$gui_specified = ''">
<xsl:for-each select="/filter/filter-itk/parameters/param">
         set $this-<xsl:value-of select="name"/> 0 
</xsl:for-each>
</xsl:when>

<xsl:otherwise>
<xsl:for-each select="/filter/filter-gui/parameters/param">
         set $this-<xsl:value-of select="@name"/><xsl:text> </xsl:text><xsl:value-of select="default"/></xsl:for-each>
</xsl:otherwise>

</xsl:choose>
    }
<!-- Create ui method only if there are parameters for
     a gui.
 -->
<xsl:choose>
<xsl:when test="$no_params = ''"> <!-- no ui method --></xsl:when>

<xsl:otherwise>
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }

        toplevel $w
<!-- Create parameter widgets.  If no gui was specified 
     have everything default to a text entry
-->  
<xsl:choose>
<!-- text-entry -->
<xsl:when test="$gui_specified = ''">
<xsl:for-each select="/filter/filter-itk/parameters/param">
<xsl:variable name="gui"><xsl:value-of select="gui"/></xsl:variable>
<xsl:call-template name="create_text_entry">
  </xsl:call-template>
</xsl:for-each>
</xsl:when>

<!-- A specific gui has been defined in the gui xml file.
     Each widget call needs certain parameters to create
     its gui and each gui will be created in its own frame.
 -->
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

<!-- Every gui must have an execute and close button -->
<xsl:call-template name="execute_and_close_buttons"/>
<xsl:text>
    }
</xsl:text>
</xsl:otherwise>
</xsl:choose>
}
</xsl:template>




<!-- ==================================================== -->
<!-- ================= WIDGET FUNCTIONS ================= -->
<!-- ==================================================== -->

<!-- Execute and Close Buttons -->
<xsl:template name="execute_and_close_buttons">
<xsl:text disable-output-escaping="yes">        
        frame $w.buttons
        button $w.buttons.execute -text &quot;Execute&quot; -command &quot;$this-c needexecute&quot;
        button $w.buttons.close -text &quot;Close&quot; -command &quot;destroy $w&quot;
        pack $w.buttons.execute $w.buttons.close -side left
	pack $w.buttons -side top -padx 5 -pady 5
</xsl:text>
</xsl:template>


<!-- Text entry gui -->
<xsl:template name="create_text_entry">
<xsl:variable name="widget">entry</xsl:variable>
<xsl:choose>
<xsl:when test="$gui_specified = ''">
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
<!--   -->
<xsl:otherwise>
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="@name"/>
</xsl:variable>
        frame <xsl:value-of select="$path"/>
        label <xsl:value-of select="$path"/><xsl:text disable-output-escaping="yes">.label -text &quot;</xsl:text><xsl:value-of select="@name"/><xsl:text disable-output-escaping="yes">&quot;
        </xsl:text>
<xsl:value-of select="$widget"/><xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/>
<xsl:text> -textvariable $this-</xsl:text><xsl:value-of select="@name"/><xsl:text>
        pack </xsl:text><xsl:value-of select="$path"/><xsl:text>.label </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> -side left
        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>
</xsl:otherwise>
</xsl:choose>
</xsl:template>

<!-- Scrollbar gui. A scrollbar must have a min and max. -->
<xsl:template name="create_scrollbar">
  <xsl:param name="from"/>
  <xsl:param name="to"/>

<xsl:variable name="widget">scale</xsl:variable>
<xsl:variable name="path"><xsl:text>$w.</xsl:text><xsl:value-of select="@name"/>
</xsl:variable>
        frame <xsl:value-of select="$path"/><xsl:text>
        </xsl:text>
<xsl:value-of select="$widget"/>
<xsl:text> </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/>
<xsl:text> -label </xsl:text> &quot;<xsl:value-of select="@name"/>&quot; \
<xsl:text>           -variable $this-</xsl:text><xsl:value-of select="@name"/> \
           -from <xsl:value-of select="$from"/> -to <xsl:value-of select="$to"/> -orient horizontal<xsl:text>
        pack </xsl:text>
<xsl:value-of select="$path"/><xsl:text>.</xsl:text><xsl:value-of select="$widget"/><xsl:text> -side left
        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>	     
</xsl:template>


<!-- Checkbutton gui -->
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
        pack </xsl:text><xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>
</xsl:template>


<!-- Radiobutton gui -->
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
        pack <xsl:value-of select="$path"/>
<xsl:text>
</xsl:text>	     
</xsl:template>
</xsl:stylesheet>
