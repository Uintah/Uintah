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
-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
<xsl:output indent="no" omit-xml-declaration="yes" method="text"/>
<xsl:strip-space elements="*"/>

<xsl:template match="/">
  <xsl:text>\documentclass[notitlepage,10pt]{article}&#xA;</xsl:text>
  <xsl:text>\usepackage{graphicx}&#xA;</xsl:text>
  <xsl:text>\usepackage{html}&#xA;</xsl:text>
  <xsl:text>\usepackage{scirun-doc}&#xA;</xsl:text>
  <xsl:text>\begin{document}&#xA;</xsl:text>

  <xsl:text>\section{Summary}&#xA;</xsl:text>
  <xsl:value-of select="./component/overview/summary"/>

  <xsl:text>\section{Use}&#xA;</xsl:text>
  <xsl:apply-templates select="./component/overview/description"/>

  <xsl:text>\subsection{Inputs}&#xA;</xsl:text>
  <xsl:apply-templates select="./component/io/inputs"/>

  <xsl:text>\subsection{Outputs}&#xA;</xsl:text>
  <xsl:apply-templates select="./component/io/outputs"/>

  <xsl:text>\subsection{Gui}&#xA;</xsl:text>
  <xsl:apply-templates select="./component/gui"/>
<!-- The "Details" section doesn't seem to map to anything in the XML file
     so we will just omit it for now -->
<!-- \section{Details} -->

  <xsl:text>\section{Notes}&#xA;</xsl:text>
  <xsl:call-template name="notes"/>

  <xsl:text>\section{Credits}&#xA;</xsl:text>
  <xsl:apply-templates select="./component/overview/authors"/>  

  <xsl:text>\end{document}&#xA;</xsl:text>
</xsl:template>

<xsl:template match="description">
  <xsl:apply-templates/>  
</xsl:template>

<xsl:template match="developer">
<!-- Ignore developer stuff -->
</xsl:template>

<xsl:template match="p">
  <xsl:text/>&#xA;&#xA;<xsl:apply-templates/>&#xA;&#xA;<xsl:text/>
</xsl:template>

<xsl:template match="note">
  <xsl:text/>&#xA;&#xA;\note{<xsl:value-of select="."/>}&#xA;&#xA;<xsl:text/>
</xsl:template>

<xsl:template match="warning">
  <xsl:text/>&#xA;&#xA;\warning{<xsl:value-of select="."/>}&#xA;&#xA;<xsl:text/>
</xsl:template>

<xsl:template match="tip">
  <xsl:text/>&#xA;&#xA;\tip{<xsl:value-of select="."/>}&#xA;&#xA;<xsl:text/>
</xsl:template>

<xsl:template match="term">
  <xsl:text/>\dfn{<xsl:value-of select="."/>}<xsl:text/>
</xsl:template>

<xsl:template match="keyboard">
  <xsl:text/>\keyboard{<xsl:value-of select="."/>}<xsl:text/>
</xsl:template>

<xsl:template match="keyword">
  <xsl:text/><xsl:value-of select="."/><xsl:text/>
</xsl:template>

<xsl:template match="acronym">
  <xsl:text/>\acronym{<xsl:value-of select="."/>}<xsl:text/>
</xsl:template>

<xsl:template match="cite">
  <xsl:text/>\etitle{<xsl:value-of select="."/>}<xsl:text/>
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
    <xsl:value-of select="."/><xsl:if test="position() != last()">, </xsl:if>
  </xsl:for-each>
</xsl:template>

<xsl:template match="outputs|inputs">
  <xsl:text>&#xA;\begin{centering}&#xA;</xsl:text>
  <xsl:for-each select="port">
    <xsl:text>\begin{tabular}{|p{6cm}|p{6cm}|} \hline&#xA;</xsl:text>
    <xsl:text/>{\emph{Name:} <xsl:value-of select="name"/>}&amp;<xsl:text/>
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
      <xsl:value-of select="."/>
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
  <xsl:text>&#xA;&#xA;Each \acronym{GUI} widget is described next:\\&#xA;</xsl:text>
  <xsl:text>\begin{centering}&#xA;</xsl:text>
  <xsl:for-each select="parameter">
    <xsl:text>\begin{tabular}{|p{4cm}|p{4cm}|p{4cm}|} \hline&#xA;</xsl:text>
    <xsl:text/>{\emph{Label:} <xsl:value-of select="label"/>}&amp;<xsl:text/>
    <xsl:text/>{\emph{Widget:} <xsl:value-of select="widget"/>}&amp;<xsl:text/>
    <xsl:text/>{\emph{Data Type:} <xsl:value-of select="datatype"/>} \\ \hline&#xA;<xsl:text/>
    <xsl:text/>\multicolumn{3}{|p{12cm}|}{<xsl:apply-templates select="description"/>} \\ \hline&#xA;<xsl:text/>
    <xsl:text>\end{tabular} \\&#xA;</xsl:text>
    <xsl:text>\vspace{0.25cm}&#xA;</xsl:text>
  </xsl:for-each>
  <xsl:text>\end{centering}&#xA;</xsl:text>
  <xsl:for-each select="img">
    <xsl:text/>\centerline{\includegraphics[]{<xsl:value-of select="concat(substring-before(., '.'),'.eps')"/>}}&#xA;<xsl:text/>
  </xsl:for-each>
</xsl:template>

<xsl:template match="latex">
  <xsl:choose>
  <xsl:when test="parent::p">
    <xsl:value-of select="."/>    
  </xsl:when>
  <xsl:otherwise><xsl:text>&#xA;&#xA;</xsl:text><xsl:value-of select="."/><xsl:text>&#xA;&#xA;</xsl:text></xsl:otherwise>
  </xsl:choose>
</xsl:template>

<xsl:template name="notes">
  <xsl:text>&#xA;\begin{itemize}&#xA;</xsl:text>
  <xsl:for-each select="./component/overview/description/developer">
    <xsl:text/>\item <xsl:apply-templates/>
  </xsl:for-each>
  <xsl:text>&#xA;\end{itemize}&#xA;</xsl:text>
</xsl:template>

</xsl:stylesheet>
