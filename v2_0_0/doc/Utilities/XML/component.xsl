<?xml version="1.0"?>

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
  A stylesheet that converts the XML based module descriptions to XHTML.
  The generated XHTML is to be used with the stylesheet component.css.
-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">
  
  <xsl:include href="top_banner.xsl"/>
  <xsl:include href="bottom_banner.xsl"/>
  
  <xsl:param name="treetop"/>
  <xsl:param name="dev" select="0"/>
  
  <xsl:template match="description">
    <xsl:if test="@id">
      <a id="{@id}"/>
    </xsl:if>
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="section">
    <xsl:if test="@id">
      <a id="{@id}"/>
    </xsl:if>
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="section/title">
    <xsl:variable name="start">
      <xsl:choose>
        <xsl:when test="ancestor::net">
          <xsl:value-of select="1"/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="0"/>
        </xsl:otherwise>
      </xsl:choose>
    </xsl:variable>
    <p>
      <xsl:attribute name="class">
        <xsl:text/>secttitle<xsl:value-of select="$start + count(ancestor::section)"/>
      </xsl:attribute>
      <xsl:apply-templates/>
    </p>
  </xsl:template>

  <xsl:template match="developer">
    <xsl:if test="$dev!=0">
      <xsl:apply-templates/>
    </xsl:if>
  </xsl:template>
  
  <xsl:template match="authors">
    <xsl:text>by </xsl:text>
    <xsl:for-each select="author">
      <xsl:apply-templates/>
      <xsl:if test="position() &lt; last()">
        <xsl:text/>, <xsl:text/>
      </xsl:if>
    </xsl:for-each>
  </xsl:template>

  <xsl:template match="orderedlist">
    <ol><xsl:apply-templates/></ol>
  </xsl:template>
  
  <xsl:template match="unorderedlist">
    <ul><xsl:apply-templates/></ul>
  </xsl:template>
  
  <xsl:template match="desclist">
    <dl><xsl:apply-templates/></dl>
  </xsl:template>
  
  <xsl:template match="listitem">
    <li><xsl:apply-templates/></li>
  </xsl:template>

  <xsl:template match="desclistitem">
    <dt><xsl:apply-templates select="desclistterm"/></dt>
    <dd><xsl:apply-templates select="desclistdef"/></dd>
  </xsl:template>

  <xsl:template match="desclistterm">
    <b><xsl:value-of select="."/></b>
  </xsl:template>

  <xsl:template match="desclistdef">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="firstterm">
    <span class="firstterm"><xsl:value-of select="."/></span>
  </xsl:template>
  
  <xsl:template match="keyword">
    <span class="firstterm"><xsl:value-of select="."/></span>
  </xsl:template>
  
  <xsl:template match="abbr">
    <span class="abbr"><xsl:value-of select="."/></span>
  </xsl:template>
  
  <xsl:template match="userinput">
    <span class="userinput"><xsl:apply-templates/></span>
  </xsl:template>
  
  <xsl:template match="cite">
    <span class="cite">
      <xsl:choose>
        <xsl:when test="@url">
          <a href="{@url}"><xsl:apply-templates/></a>
        </xsl:when>
        <xsl:otherwise>
          <xsl:apply-templates/>
        </xsl:otherwise>
      </xsl:choose>
    </span>
  </xsl:template>
  
  <xsl:template match="emph">
    <span class="emph"><xsl:apply-templates/></span>
  </xsl:template>
  
  <xsl:template match="quote">
    &#147;<xsl:apply-templates/>&#148;
  </xsl:template>

  <xsl:template match="rlink">
    <a>
      <xsl:attribute name="href">
        <xsl:value-of select="$treetop"/><xsl:value-of select="@path"/>
      </xsl:attribute>
      <xsl:apply-templates/>
    </a>
  </xsl:template>
  
  <xsl:template match="slink">
    <a href="http://www.sci.utah.edu/{@path}"><xsl:apply-templates/></a>
  </xsl:template>
  
  <xsl:template match="ulink">
    <a href="{@url}"><xsl:apply-templates/></a>
  </xsl:template>
  
  <xsl:template match="xref">
    <xsl:variable name="targetnode" select="id(@target)"/>
    <xsl:choose>
      <xsl:when test="$targetnode">
        <xsl:variable name="targetnodename" select="name($targetnode)"/>
        <xsl:choose>
          <xsl:when test="$targetnodename='description'">
            <xsl:call-template name="descref">
              <xsl:with-param name="node" select="$targetnode"/>
              <xsl:with-param name="target" select="@target"/>
            </xsl:call-template>
          </xsl:when>
          <xsl:when test="$targetnodename='figure'">
            <xsl:call-template name="figref">
              <xsl:with-param name="node" select="$targetnode"/>
              <xsl:with-param name="target" select="@target"/>
            </xsl:call-template>
          </xsl:when>
          <xsl:when test="$targetnodename='net'">
            <xsl:call-template name="netref">
              <xsl:with-param name="node" select="$targetnode"/>
              <xsl:with-param name="target" select="@target"/>
            </xsl:call-template>
          </xsl:when>
          <xsl:when test="$targetnodename='section'">
            <xsl:call-template name="sectref">
              <xsl:with-param name="node" select="$targetnode"/>
              <xsl:with-param name="target" select="@target"/>
            </xsl:call-template>
          </xsl:when>
        </xsl:choose>
      </xsl:when>
      <xsl:otherwise>
        <xsl:message terminate="yes">
          <xsl:text/>For xref target=<xsl:value-of select="@target"/>: Target does not exist<xsl:text/>
        </xsl:message>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template name="descref">
    <xsl:param name="node"/>
    <xsl:param name="target"/>
    <a>
      <xsl:attribute name="href">
        <xsl:text/>#<xsl:value-of select="$target"/>
      </xsl:attribute>
      <xsl:choose>
        <xsl:when test="node()">
          <xsl:apply-templates/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:choose>
            <xsl:when test="$node/title">
              <xsl:apply-templates select="$node/title/child::node()"/>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text>this description</xsl:text>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template name="sectref">
    <xsl:param name="node"/>
    <xsl:param name="target"/>
    <a>
      <xsl:attribute name="href">
        <xsl:text/>#<xsl:value-of select="$target"/>
      </xsl:attribute>
      <xsl:choose>
        <xsl:when test="node()">
          <xsl:apply-templates/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:choose>
            <xsl:when test="$node/title">
              <xsl:apply-templates select="$node/title/child::node()"/>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text>this section</xsl:text>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template match="net/title">
    <p class="nettitle">
      <xsl:apply-templates/>
    </p>
  </xsl:template>

  <xsl:template match="net">
    <xsl:if test="@id">
      <a id="{@id}"/>
    </xsl:if>
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template name="netref">
    <xsl:param name="node"/>
    <xsl:param name="target"/>
    <a>
      <xsl:attribute name="href">
        <xsl:text/>#<xsl:value-of select="$target"/>
      </xsl:attribute>
      <xsl:choose>
        <xsl:when test="node()">
          <xsl:apply-templates/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:choose>
            <xsl:when test="$node/title">
              <xsl:apply-templates select="$node/title/child::node()"/>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text>A network example</xsl:text>
            </xsl:otherwise>
          </xsl:choose>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template name="figref">
    <xsl:param name="node"/>
    <xsl:param name="target"/>
    <a>
      <xsl:attribute name="href">
        <xsl:text/>#<xsl:value-of select="$target"/>
      </xsl:attribute>
      <xsl:choose>
        <xsl:when test="node()">
          <xsl:apply-templates/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text/>Figure <xsl:apply-templates select="$node" mode="figref"/>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template match="examplesr">
    <a>
      <xsl:attribute name="href">
        <xsl:value-of select="."/>
      </xsl:attribute>
      <p class="subtitle">
        Example Network
      </p>
    </a>
  </xsl:template>
  
  <xsl:template match="modref">
    <a>
      <xsl:attribute name="href">
        <xsl:value-of select="$treetop"/>src/<xsl:text/>
        <xsl:if test="@package != 'SCIRun'">
          <xsl:text/>Packages/<xsl:value-of select="@package"/>/<xsl:text/>
        </xsl:if>
        <xsl:text/>DataFlow/XML/<xsl:value-of select="@name"/>.html<xsl:text/>
      </xsl:attribute>
      <xsl:choose>
        <xsl:when test="node()">
          <xsl:value-of select="."/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:text/><xsl:value-of select="@name"/>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template match="email">
    <a href="mailto:{@addr}">
      <xsl:choose>
        <xsl:when test="node()">
          <xsl:value-of select="."/>
        </xsl:when>
        <xsl:otherwise>
          <xsl:value-of select="@addr"/>
        </xsl:otherwise>
      </xsl:choose>
    </a>
  </xsl:template>

  <xsl:template match="parameter">
    <tr>
      <td><xsl:value-of select="widget"/></td>
      <td><xsl:value-of select="label"/></td>
      <td><xsl:apply-templates select="description"/></td>
    </tr>
  </xsl:template>
  
  <xsl:template match="img">
    <center>
      <img vspace="20">
        <xsl:attribute name="src">
          <xsl:value-of select="."/>
        </xsl:attribute>
      </img>
    </center>
  </xsl:template>
  
  <xsl:template match="port">
    <xsl:param name="last"/>
    <tr>
      <td>
        <xsl:choose>
          <xsl:when test="$last='last'">Dynamic Port</xsl:when>
          <xsl:otherwise>Port</xsl:otherwise>
        </xsl:choose>
      </td>
      <td><xsl:value-of select="datatype"/></td>
      <td><xsl:value-of select="name"/></td>
      <td><xsl:apply-templates select="description"/></td>
    </tr>
  </xsl:template>
  
  <xsl:template match="file">
    <tr>
      <td>File</td>
      <td><xsl:value-of select="datatype"/></td>
      <td></td>
      <td><xsl:apply-templates select="description"/></td>
    </tr>
    
  </xsl:template>
  
  <xsl:template match="device">
    <tr>
      <td>Device</td>
      <td></td>
      <td><xsl:value-of select="devicename"/></td>
      <td><xsl:apply-templates select="description"/></td>
    </tr>
  </xsl:template>
  
  <xsl:template match="p">
    <p><xsl:apply-templates/></p>
  </xsl:template>
  
  <xsl:template match="note">
    <div class="admon">
      <p class="admontitle">Note</p>
      <xsl:apply-templates/>
    </div>
  </xsl:template>
  
  <xsl:template match="tip">
    <div class="admon">
      <p class="admontitle">Tip</p>
      <xsl:apply-templates/>
    </div>
  </xsl:template>
  
  <xsl:template match="warning">
    <div class="admon">
      <p class="admontitle">Warning</p>
      <xsl:apply-templates/>
    </div>
  </xsl:template>

  <xsl:template match="figure">
    <blockquote>
      <p align="left">
        <img id="{@id}" src="{img}"/>
      </p>
      <p align="left">
        <strong>Figure <xsl:number level="any"/>: <xsl:apply-templates select="caption"/></strong>
      </p>
    </blockquote>
  </xsl:template>
  
  <xsl:template match="caption">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="figure" mode="figref">
    <xsl:number level="any"/>
  </xsl:template>

  <xsl:template match="pre">
    <pre>
      <xsl:apply-templates/>
    </pre>
  </xsl:template>

  <xsl:template match="inputs">
    <xsl:variable name="dynamic">
      <xsl:value-of select="@lastportdynamic"/>
    </xsl:variable> 
    <xsl:for-each select="port">
      <xsl:variable name="portnum"><xsl:number/></xsl:variable>
      <xsl:variable name="last">
        <xsl:if test="$portnum=last() and $dynamic='yes'">last</xsl:if>
      </xsl:variable>
      <xsl:apply-templates select=".">
        <xsl:with-param name="last">
          <xsl:value-of select="$last"/>
        </xsl:with-param>
      </xsl:apply-templates>
    </xsl:for-each>
    <xsl:for-each select="file">
      <xsl:apply-templates select="."/>
    </xsl:for-each>
    <xsl:for-each select="device">
      <xsl:apply-templates select="."/>
    </xsl:for-each>
  </xsl:template>
  
  <xsl:template match="outputs">
    <xsl:for-each select="port">
      <xsl:apply-templates select="."/>
    </xsl:for-each>
    <xsl:for-each select="file">
      <xsl:apply-templates select="."/>
    </xsl:for-each>
    <xsl:for-each select="device">
      <xsl:apply-templates select="."/>
    </xsl:for-each>
  </xsl:template>
  
  <xsl:template match="gui">
    <xsl:apply-templates select="description"/>
    <xsl:apply-templates select="img"/>
    <p>
      <table class="gui" cellspacing="0" border="1" width="100%" cellpadding="2">
        <tr><th colspan="3">Descriptions of GUI Controls (Widgets)</th></tr>
        <tr><th>Widget</th><th>Label</th><th>Description</th></tr>
        <xsl:for-each select="parameter">
          <xsl:apply-templates select="."/>
        </xsl:for-each>
      </table>
    </p>
  </xsl:template>
  
  <xsl:template match="/component">
    <html>
      <head>
        <title><xsl:value-of select="@name" /></title>
        <link rel="stylesheet" type="text/css">
          <xsl:attribute name="href">
            <xsl:value-of select="concat($treetop,'doc/Utilities/HTML/component.css')" />
            </xsl:attribute>
        </link>
          
      </head>
      <body>
        
        <xsl:call-template name="top_banner"/>
        
        <div class="title"><xsl:value-of select="@name"/></div>
        <div class="subtitle">Category: <xsl:value-of select="@category"/></div>
        <div class="authors"><xsl:apply-templates select="overview/authors"/></div>

        <xsl:apply-templates select="overview/examplesr"/>
        
        <p class="head">Summary</p>
        
        <p>
          <xsl:apply-templates select="overview/summary"/>
        </p>
        
        <p class="head">Description</p>
        
        <p>
          <xsl:apply-templates select="overview/description"/>
        </p>
        
        <p class="head">I/O</p>
        
        <p>
          <table class="io" cellspacing="0" border="1" width="100%" cellpadding="2">
            <tr>
              <th colspan="5" align="center"><xsl:value-of select="@name"/>'s Inputs</th>
            </tr>
            <tr>
              <th>I/O Type</th>
              <th>Datatype</th>
              <th>Name</th>
              <th>Description</th>
            </tr>
            <xsl:apply-templates select="io/inputs"/>
            
          </table>
        </p>
        <p>
          <table class="io" cellspacing="0" border="1" width="100%" cellpadding="2">
            <tr>
              <th colspan="5" align="center"><xsl:value-of select="@name"/>'s Outputs</th>
            </tr>
            <tr>
              <th>I/O Type</th>
              <th>Datatype</th>
              <th>Name</th>
              <th>Description</th>
            </tr>
            <xsl:apply-templates select="io/outputs"/>
          </table>
        </p>
        
        <xsl:if test="gui">
          <p class="head">GUI</p>
          <xsl:apply-templates select="gui"/>
        </xsl:if>

        <xsl:if test="nets">
          <p class="head">Example Networks</p>
          <xsl:apply-templates select="nets"/>
        </xsl:if>
        
        <xsl:call-template name="bottom_banner"/>
        
      </body>
    </html>
  </xsl:template>
</xsl:stylesheet>
