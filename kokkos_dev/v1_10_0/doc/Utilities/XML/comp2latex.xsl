<?xml version="1.0" encoding="iso-8859-1"?>

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

<!--
This stylesheet does its best to convert a component.xml file to the 
latex version of the module description. 

In particular, it maps xml:summary to latex:summary;
xml:(description,io,gui) to latex:use; xml:authors to latex:credits; xml
developer notes in the overview/description to latex:notes.  

Normally the stylesheet will generate a standalone latex document.  Setting
the parameter "frag" to 1 will cause the stylesheet to generate a 
section level latex fragment that may be \input by another latex document.

When in frag mode you will want to adjust the relative path to images (and
other things?) so that the img path is relative to the location of the tex
document inputting the fragment rather than being relative to the location
of the xml file.  To do this set the parameter "relpath" to be the relative
path from the location of the tex document to the location of the xml
document.

Note: Obviously, this stylesheet may generate bad latex if the xml input
does not conform to component.dtd.  Validate the xml input first!  

-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  <xsl:output indent="no" omit-xml-declaration="yes" method="text"/>
  <xsl:strip-space elements="*"/>

  <xsl:param name="frag" select="0"/>

  <xsl:param name="relpath" select="''"/>

  <xsl:variable name="section">
    <xsl:choose>
      <xsl:when test="$frag=0">
        <xsl:text>section</xsl:text>
      </xsl:when>
      <xsl:when test="$frag=1">
        <xsl:text>subsection</xsl:text>
      </xsl:when>
    </xsl:choose>
  </xsl:variable>

  <xsl:variable name="subsection">
    <xsl:choose>
      <xsl:when test="$frag=0">
        <xsl:text>subsection</xsl:text>
      </xsl:when>
      <xsl:when test="$frag=1">
        <xsl:text>subsubsection*</xsl:text>
      </xsl:when>
    </xsl:choose>
  </xsl:variable>

  <xsl:variable name="modname">
    <xsl:value-of select="//component/@name"/>
  </xsl:variable>

  <xsl:variable name="category">
    <xsl:value-of select="//component/@category"/>
  </xsl:variable>
  
  <xsl:template match="/">
    <xsl:choose>
      <!-- Transform into a doc fragment -->
      <xsl:when test="$frag=1">
        <xsl:text/>\<xsl:value-of select="$section"/>{<xsl:value-of select="$modname"/>}&#xA;\index{<xsl:value-of select="$modname"/>}&#xA;<xsl:text/>
      </xsl:when>
      <!-- Transform into a complete document. -->
      <xsl:when test="$frag=0">
        <xsl:text>\documentclass[notitlepage,10pt]{article}&#xA;</xsl:text>
        <xsl:text>\usepackage{graphicx}&#xA;</xsl:text>
        <xsl:text>\usepackage{html}&#xA;</xsl:text>
        <xsl:text>%begin{latexonly}&#xA;</xsl:text>
        <xsl:text>\usepackage{scirun-doc}&#xA;</xsl:text>
        <xsl:text>%end{latexonly}&#xA;</xsl:text>
        <xsl:text>\begin{htmlonly}&#xA;</xsl:text>
        <xsl:text>\input{scirun-doc.tex}&#xA;</xsl:text>
        <xsl:text>\end{htmlonly}&#xA;</xsl:text>
        <xsl:text/>\title{<xsl:value-of select="$modname"/>}&#xA;<xsl:text/>
        <xsl:text>\begin{document}&#xA;</xsl:text>
        <xsl:text>\maketitle &#xA;</xsl:text>
      </xsl:when>
    </xsl:choose>
    
    <!-- Transform major sections but only if they are non-empty -->
    <xsl:if test="./component/overview/summary/child::*">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Summary}&#xA;<xsl:text/>
      <xsl:value-of select="./component/overview/summary"/>
    </xsl:if>
    
    <xsl:if test="./component/overview/description/child::*">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Use}&#xA;<xsl:text/>
      <xsl:apply-templates select="./component/overview/description"/>
    </xsl:if>
    
    <xsl:if test="./component/io/inputs/child::*">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Inputs}&#xA;<xsl:text/>
      <xsl:apply-templates select="./component/io/inputs"/>
    </xsl:if>
    
    <xsl:if test="./component/io/outputs/child::*">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Outputs}&#xA;<xsl:text/>
      <xsl:apply-templates select="./component/io/outputs"/>
    </xsl:if>
    
    <xsl:if test="./component/gui/child::*">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Gui}&#xA;<xsl:text/>
      <xsl:apply-templates select="./component/gui"/>
    </xsl:if>

    <!-- The "Details" section doesn't seem to map to anything in the XML file
         so we will just omit it for now -->
    <!-- \section{Details} -->
    
    <xsl:call-template name="notes"/>
    
    <xsl:if test="./component/overview/authors/child::*">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Credits}&#xA;<xsl:text/>
      <xsl:apply-templates select="./component/overview/authors"/>  
    </xsl:if>
    
    <!-- Terminate doc frag or complete document correctly -->
    <xsl:choose>
      <xsl:when test="$frag=0">
        <xsl:text>&#xA;\end{document}&#xA;</xsl:text>
      </xsl:when>
      <xsl:when test="$frag=1">
        <xsl:text>&#xA;&#xA;</xsl:text>
      </xsl:when>
    </xsl:choose>

  </xsl:template>
  
  <!-- 
       Templates for transforming component.xml elements.  Mostly obvious
       except where noted.  I use lots of xsl:text elements for controlling
       line-endings in order to get latex to do the right thing. 
   -->

  <xsl:template match="description">
    <xsl:apply-templates/>  
  </xsl:template>
  
  <xsl:template match="developer">
    <!-- Ignore developer stuff -->
  </xsl:template>
  
  <xsl:template match="p">
    <xsl:text>&#xA;&#xA;</xsl:text><xsl:apply-templates/><xsl:text>&#xA;&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="note">
    <xsl:text>&#xA;&#xA;</xsl:text>{\slshape Note: <xsl:apply-templates/>}<xsl:text>&#xA;&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="warning">
    <xsl:text>&#xA;&#xA;</xsl:text>{\bfshape Warning: <xsl:apply-templates/>}<xsl:text>&#xA;&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="tip">
    <xsl:text>&#xA;&#xA;</xsl:text>{\slshape Tip: <xsl:apply-templates/>}<xsl:text>&#xA;&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="term">
    <xsl:text/>\dfn{<xsl:apply-templates/>}<xsl:text/>
  </xsl:template>
  
  <xsl:template match="keyboard">
    <xsl:text/>\keyboard{<xsl:apply-templates/>}<xsl:text/>
  </xsl:template>
  
  <xsl:template match="keyword">
    <xsl:text/><xsl:apply-templates/><xsl:text/>
  </xsl:template>
  
  <xsl:template match="acronym">
    <xsl:text/>\acronym{<xsl:apply-templates/>}<xsl:text/>
  </xsl:template>
  
  <xsl:template match="cite">
    <xsl:text/>\etitle{<xsl:apply-templates/>}<xsl:text/>
  </xsl:template>
  
  <xsl:template match="orderedlist">
    <xsl:text>&#xA;\begin{enumerate}&#xA;</xsl:text>
    <xsl:apply-templates/>
    <xsl:text>&#xA;\end{enumerate}&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="unorderedlist">
    <xsl:text>&#xA;\begin{itemize}&#xA;</xsl:text>
    <xsl:apply-templates/>
    <xsl:text>&#xA;\end{itemize}&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="desclist">
    <xsl:text>&#xA;\begin{description}&#xA;</xsl:text>
    <xsl:apply-templates/>
    <xsl:text>&#xA;\end{description}&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="listitem">
    <xsl:text/>&#xA;\item <xsl:apply-templates/><xsl:text/>
  </xsl:template>
  
  <xsl:template match="desclistitem">
    <xsl:text/>&#xA;\item[<xsl:value-of select="desclistterm"/>] <xsl:text/>
    <xsl:apply-templates select="desclistdef"/>
  </xsl:template>
  
  <xsl:template match="desclistdef">
    <xsl:apply-templates/>
  </xsl:template>
  
  <xsl:template match="authors">
    <xsl:for-each select="author">
      <xsl:apply-templates/><xsl:if test="position() != last()">, </xsl:if>
    </xsl:for-each>
  </xsl:template>
  
  <xsl:template match="outputs|inputs">
    <xsl:text>&#xA;\begin{centering}&#xA;</xsl:text>
    <xsl:for-each select="port">
      <xsl:text>\begin{tabular}{|p{6cm}|p{6cm}|} \hline&#xA;</xsl:text>
      <xsl:text/>{\emph{Name:} <xsl:apply-templates select="name"/>}&amp;<xsl:text/>
      <xsl:text>{\emph{Type:} </xsl:text>
      <xsl:if test="position()=last() and ancestor::inputs and ancestor::inputs[@lastportdynamic='yes']">
        <xsl:text>Dynamic </xsl:text>
      </xsl:if>
      <xsl:text>Port}\\ \hline&#xA;</xsl:text>
      <xsl:text/>\multicolumn{2}{|p{12cm}|}{<xsl:apply-templates select="description"/>}\\ \hline&#xA;<xsl:text/>
      <xsl:text>\multicolumn{2}{|p{12cm}|}{</xsl:text>
      <xsl:choose>
        <xsl:when test="ancestor::inputs">\emph{Upstream Module(s):} </xsl:when>
        <xsl:when test="ancestor::outputs">\emph{Downstream Module(s):} </xsl:when>
      </xsl:choose>
      <xsl:for-each select="componentname">
        <xsl:apply-templates/>
        <xsl:if test="position() != last()">,  </xsl:if>
        <xsl:text> </xsl:text>
      </xsl:for-each>}\\ \hline&#xA;<xsl:text/>
      <xsl:text>\end{tabular} \\&#xA;</xsl:text>
      <xsl:text>\vspace{0.25cm}&#xA;</xsl:text>
    </xsl:for-each>
    <xsl:for-each select="file">
      <xsl:text>\begin{tabular}{|p{12cm}|} \hline&#xA;</xsl:text>
      <xsl:text>{\emph{Type:} File} \\ \hline&#xA;</xsl:text>
      <xsl:text/>{<xsl:apply-templates select="description"/>} \\ \hline&#xA;<xsl:text/>
      <xsl:text>\end{tabular} \\&#xA;</xsl:text>
      <xsl:text>\vspace{0.25cm}&#xA;</xsl:text>
    </xsl:for-each>
    <xsl:for-each select="device">
      <xsl:text>\begin{tabular}{|p{12cm}|} \hline&#xA;</xsl:text>
      <xsl:text>{\emph{Type:} Device} \\ \hline&#xA;</xsl:text>
      <xsl:text/>{<xsl:apply-templates select="description"/>} \\ \hline&#xA;<xsl:text/>
      <xsl:text>\end{tabular} \\&#xA;</xsl:text>
      <xsl:text>\vspace{0.25cm}&#xA;</xsl:text>
    </xsl:for-each>
    <xsl:text>\end{centering}&#xA;</xsl:text>
  </xsl:template>
  
  <xsl:template match="gui">
    <xsl:apply-templates select="description"/>
    <xsl:text>&#xA;&#xA;Each \acronym{GUI} widget is described next.  See also Figure</xsl:text>
    <xsl:if test="count(img)&gt;1">s</xsl:if>~<xsl:text/>
    <xsl:for-each select="img">
      <xsl:variable name="fig"><xsl:value-of select="$modname"/>Fig<xsl:number count='img' format='A'/></xsl:variable>
      <xsl:text/>\ref{fig:<xsl:value-of select="$fig"/>}<xsl:text/>
      <xsl:choose>
        <xsl:when test="position()+1=last()">
          <xsl:text> and </xsl:text>
        </xsl:when>
        <xsl:when test="position() != last()">
          <xsl:text>, </xsl:text>
        </xsl:when>
      </xsl:choose>
    </xsl:for-each>
    <xsl:text>.&#xA;</xsl:text>
    <xsl:text>\begin{centering}&#xA;</xsl:text>
    <xsl:for-each select="parameter">
      <xsl:text>\begin{tabular}{|p{4cm}|p{4cm}|p{4cm}|} \hline&#xA;</xsl:text>
      <xsl:text/>{\emph{Label:} <xsl:apply-templates select="label"/>}&amp;<xsl:text/>
      <xsl:text/>{\emph{Widget:} <xsl:apply-templates select="widget"/>}&amp;<xsl:text/>
      <xsl:text/>{\emph{Data Type:} <xsl:apply-templates select="datatype"/>} \\ \hline&#xA;<xsl:text/>
      <xsl:text/>\multicolumn{3}{|p{12cm}|}{<xsl:apply-templates select="description"/>} \\ \hline&#xA;<xsl:text/>
      <xsl:text>\end{tabular} \\&#xA;</xsl:text>
      <xsl:text>\vspace{0.25cm}&#xA;</xsl:text>
    </xsl:for-each>
    <xsl:text>\end{centering}&#xA;</xsl:text>
    
    <!-- Generate figure commands -->
    <xsl:for-each select="img">
      <xsl:variable name="figcmd">\<xsl:value-of select="$modname"/>Fig<xsl:number count='img' format='A'/></xsl:variable>
      <xsl:text>%begin{latexonly}&#xA;</xsl:text>
      <xsl:text/>\newcommand{<xsl:value-of select="$figcmd"/>}%&#xA;<xsl:text/>
      <xsl:variable name="basename">
        <xsl:choose>
          <xsl:when test="contains(.,'.jpg')">
            <xsl:value-of select="substring-before(., '.jpg')"/>
          </xsl:when>
          <xsl:when test="contains(.,'.gif')">
            <xsl:value-of select="substring-before(., '.gif')"/>
          </xsl:when>
          <xsl:when test="contains(.,'.tiff')">
            <xsl:value-of select="substring-before(., '.tiff')"/>
          </xsl:when>
          <xsl:when test="contains(.,'.png')">
            <xsl:value-of select="substring-before(., '.png')"/>
          </xsl:when>
        </xsl:choose>
      </xsl:variable>
      <xsl:text/>{\centerline{\includegraphics[]{<xsl:value-of select="concat($relpath,concat($basename,'.eps'))"/>}}}&#xA;<xsl:text/>
      <xsl:text>%end{latexonly}&#xA;</xsl:text>
      <xsl:text>\begin{htmlonly}&#xA;</xsl:text>
      <xsl:text/>\newcommand{<xsl:value-of select="$figcmd"/>}{%&#xA;<xsl:text/>
      <xsl:text/>\htmladdimg[]{../<xsl:value-of select="concat($relpath, .)"/>}}&#xA;<xsl:text/>
      <xsl:text>\end{htmlonly}&#xA;</xsl:text>
    </xsl:for-each>
    
    <!-- Generate figure calling commands -->
    <xsl:for-each select="img">
      <xsl:text>\begin{figure}[htb]&#xA;</xsl:text>
      <xsl:text>\begin{makeimage}&#xA;</xsl:text>
      <xsl:text>\end{makeimage}&#xA;</xsl:text>
      <xsl:variable name="figcmd"><xsl:value-of select="$modname"/>Fig<xsl:number count='img' format='A'/></xsl:variable>
      <xsl:text/>\<xsl:value-of select="$figcmd"/><xsl:text>&#xA;</xsl:text>
      <xsl:variable name="captionnode" select="following-sibling::*[1][self::caption]"/>
      <xsl:choose>
        <xsl:when test="$captionnode">
          <xsl:text/>\caption{<xsl:value-of select="$captionnode"/>}&#xA;<xsl:text/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text/>\caption{GUI for module <xsl:value-of select="/component/@name"/>}&#xA;<xsl:text/>
        </xsl:otherwise>
      </xsl:choose>
      <xsl:text/>\label{fig:<xsl:value-of select="$figcmd"/>}&#xA;<xsl:text/>
      <xsl:text>\end{figure}&#xA;</xsl:text>
    </xsl:for-each>
  </xsl:template>
  
  <xsl:template match="latex">
    <xsl:choose>
      <xsl:when test="parent::p">
        <xsl:value-of select="."/>    
      </xsl:when>
      <xsl:otherwise>
        <xsl:text>&#xA;&#xA;</xsl:text><xsl:value-of select="."/><xsl:text>&#xA;&#xA;</xsl:text>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- Scour latex specials from text nodes. -->
  <xsl:template match="text()">
    <xsl:if test="not(parent::latex)">
      <xsl:call-template name="filter-specials">
        <xsl:with-param name="input" select="."/>
      </xsl:call-template>
    </xsl:if>
  </xsl:template>
    
  <!-- Callable templates follow -->
  
  <xsl:template name="notes">
    <xsl:variable name="notes" select="./component/overview/description/developer"/>
    <xsl:if test="$notes">
      <xsl:text/>\<xsl:value-of select="$subsection"/>{Notes}&#xA;<xsl:text/>
      <xsl:text>&#xA;\begin{itemize}&#xA;</xsl:text>
      <xsl:for-each select="$notes">
        <xsl:text/>\item <xsl:apply-templates/>
      </xsl:for-each>
      <xsl:text>&#xA;\end{itemize}&#xA;</xsl:text>
    </xsl:if>
  </xsl:template>
  
  <!-- Replace in "source" all occurrences of "this" with "that" -->
  <xsl:template name="replace_all">
    <xsl:param name="source"/>
    <xsl:param name="this"/>
    <xsl:param name="that"/>
    <xsl:choose>
      <xsl:when test="contains($source, $this)">
        <xsl:value-of select="substring-before($source, $this)"/>
        <xsl:value-of select="$that"/>
        <xsl:call-template name="replace_all">
          <xsl:with-param name="source" select="substring-after($source, $this)"/>
          <xsl:with-param name="this" select="$this"/>
          <xsl:with-param name="that" select="$that"/>
        </xsl:call-template>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="$source"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <!-- Recode latex special symbols into something latex will like -->
  <xsl:template name="filter-specials">
    <xsl:param name="input"/>
    <xsl:variable name="r1">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$input"/>
        <xsl:with-param name="this" select="'\'"/>
        <xsl:with-param name="that" select="'\textbackslash{}'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r2">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r1"/>
        <xsl:with-param name="this" select="'$'"/>
        <xsl:with-param name="that" select="'\$'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r3">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r2"/>
        <xsl:with-param name="this" select="'&amp;'"/>
        <xsl:with-param name="that" select="'\&amp;'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r4">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r3"/>
        <xsl:with-param name="this" select="'%'"/>
        <xsl:with-param name="that" select="'\%'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r5">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r4"/>
        <xsl:with-param name="this" select="'#'"/>
        <xsl:with-param name="that" select="'\#'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r6">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r5"/>
        <xsl:with-param name="this" select="'_'"/>
        <xsl:with-param name="that" select="'\_'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r7">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r6"/>
        <xsl:with-param name="this" select="'{'"/>
        <xsl:with-param name="that" select="'\{'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r8">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r7"/>
        <xsl:with-param name="this" select="'}'"/>
        <xsl:with-param name="that" select="'\}'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r9">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r8"/>
        <xsl:with-param name="this" select="'~'"/>
        <xsl:with-param name="that" select="'\textasciitilde{}'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r10">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r9"/>
        <xsl:with-param name="this" select="'^'"/>
        <xsl:with-param name="that" select="'\textasciicircum{}'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r11">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r10"/>
        <xsl:with-param name="this" select="'&lt;'"/>
        <xsl:with-param name="that" select="'\textless{}'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:variable name="r12">
      <xsl:call-template name="replace_all">
        <xsl:with-param name="source" select="$r11"/>
        <xsl:with-param name="this" select="'&gt;'"/>
        <xsl:with-param name="that" select="'\textgreater{}'"/>
      </xsl:call-template>
    </xsl:variable>
    <xsl:value-of select="$r12"/>
  </xsl:template>

</xsl:stylesheet>
