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

This stylesheet converts the XML based module descriptions to HTML.
The generated HTML is to be used with the stylesheet component.css.

-->

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

  <xsl:include href="top_banner.xsl"/>
  <xsl:include href="bottom_banner.xsl"/>

  <xsl:param name="treetop"/>

  <xsl:template match="developer" mode="dev">
    <xsl:apply-templates/>
  </xsl:template>

  <xsl:template match="orderedlist">
    <ol><xsl:apply-templates/></ol>
  </xsl:template>

  <xsl:template match="unorderedlist">
    <ul><xsl:apply-templates/></ul>
  </xsl:template>

  <xsl:template match="desclist">
    <ul><p><xsl:apply-templates/></p></ul>
  </xsl:template>

  <xsl:template match="term">
    <span class="term"><xsl:value-of select="."/></span>
  </xsl:template>

  <xsl:template match="keyboard">
    <span class="keyboard"><xsl:apply-templates/></span>
  </xsl:template>

  <xsl:template match="cite">
    <span class="cite"><xsl:apply-templates/></span>
  </xsl:template>

  <xsl:template match="rlink">
    <a>
      <xsl:attribute name="href">
        <xsl:value-of select="@path"/>
      </xsl:attribute>
      <xsl:apply-templates/>
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
  
  <xsl:template match="developer">
  </xsl:template>
  
  <xsl:template match="parameter">
    <tr>
      <td><xsl:value-of select="./widget"/></td>
      <td><xsl:value-of select="./label"/></td>
      <td><xsl:apply-templates select="./description"/></td>
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
      <td><xsl:value-of select="./datatype"/></td>
      <td><xsl:value-of select="./name"/></td>
      <td><xsl:apply-templates select="./description"/></td>
    </tr>
  </xsl:template>
  
  <xsl:template match="file">
    <tr>
      <td>File</td>
      <td><xsl:value-of select="./datatype"/></td>
      <td></td>
      <td><xsl:apply-templates select="./description"/></td>
    </tr>
    
  </xsl:template>
  
  <xsl:template match="device">
    <tr>
      <td>Device</td>
      <td></td>
      <td><xsl:value-of select="./devicename"/></td>
      <td><xsl:apply-templates select="./description"/></td>
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
          
          <xsl:apply-templates select="./overview/examplesr"/>
          
          <p class="head">Summary</p>
          
          <p>
            <xsl:for-each select="./overview/summary">
              <xsl:apply-templates/>
            </xsl:for-each>
          </p>
          
          <p class="head">Description</p>
          
          <p>
            <xsl:for-each select="./overview/description">
              <xsl:apply-templates/>
            </xsl:for-each>
          </p>
          
          <p class="head">I/O</p>
          
          <p><table class="io" cellspacing="0" border="1" width="100%" cellpadding="2">
          <tr>
            <th colspan="5" align="center"><xsl:value-of select="@name"/>'s Inputs</th>
          </tr>
          <tr>
            <th>I/O Type</th>
            <th>Datatype</th>
            <th>Name</th>
            <th>Description</th>
          </tr>
          <xsl:for-each select="./io/inputs">
            <xsl:variable name="dynamic">
              <xsl:value-of select="@lastportdynamic"/>
            </xsl:variable> 
            <xsl:for-each select="./port">
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
            <xsl:for-each select="./file">
              <xsl:apply-templates select="."/>
            </xsl:for-each>
            <xsl:for-each select="./device">
              <xsl:apply-templates select="."/>
            </xsl:for-each>
          </xsl:for-each>
        </table></p>
        <p><table class="io" cellspacing="0" border="1" width="100%" cellpadding="2">
        <tr>
          <th colspan="5" align="center"><xsl:value-of select="@name"/>'s Outputs</th>
        </tr>
        <tr>
          <th>I/O Type</th>
          <th>Datatype</th>
          <th>Name</th>
          <th>Description</th>
        </tr>
        <xsl:for-each select="./io/outputs">
          <xsl:for-each select="./port">
            <xsl:apply-templates select="."/>
          </xsl:for-each>
          <xsl:for-each select="./file">
            <xsl:apply-templates select="."/>
          </xsl:for-each>
          <xsl:for-each select="./device">
            <xsl:apply-templates select="."/>
          </xsl:for-each>
        </xsl:for-each>
      </table></p>
      
      <xsl:for-each select="./gui">
        <p class="head">GUI</p>
        <xsl:apply-templates select="./description"/>
        <xsl:apply-templates select="./img"/>
        <p>
          <table class="gui" cellspacing="0" border="1" width="100%" cellpadding="2">
            <tr><th colspan="3">Descriptions of GUI Controls (Widgets)</th></tr>
            <tr><th>Widget</th><th>Label</th><th>Description</th></tr>
            <xsl:for-each select="./parameter">
              <xsl:apply-templates select="."/>
            </xsl:for-each>
          </table>
        </p>
      </xsl:for-each>
      
      <xsl:call-template name="bottom_banner"/>
      
    </body>
  </html>
</xsl:template>
</xsl:stylesheet>
