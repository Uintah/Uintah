<?xml version='1.0'?>

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

SCI modifications to qandaset processing.

The unmodified DocBook stylesheets produce horrid looking faqs (what
was Norm thinking?!).  This code is an attempt to fix that.

Templates in this file override templates from qandaset.xsl and
block.xsl.  And there are some new templates too.  This file is
"included" by "srdocbook-common.xsl".

Note: This is a fairly uneducated hack.  Basically, when the
"defaultlabel" attribute on the "qandaset" element is set to one of
"none", "number", or "qanda" then this code generates something
decent, otherwise probably not.  In particular, the use of
'defaultlabel="label"' is not supported.  Leaving out the
"defaultlabel" attribute is not really supported either.  The use of
the qanddiv element seems to work.

-->

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform"
                xmlns:doc="http://nwalsh.com/xsl/documentation/1.0"
                exclude-result-prefixes="doc"
                version='1.0'>

  <xsl:template match="qandaset">
    <xsl:variable name="title" select="title"/>
    <xsl:variable name="preamble" select="*[name(.) != 'title'
                                            and name(.) != 'titleabbrev'
                                            and name(.) != 'qandadiv'
                                            and name(.) != 'qandaentry']"/>
    <xsl:variable name="label-width">
      <xsl:call-template name="dbhtml-attribute">
        <xsl:with-param name="pis"
                        select="processing-instruction('dbhtml')"/>
        <xsl:with-param name="attribute" select="'label-width'"/>
      </xsl:call-template>
    </xsl:variable>

    <xsl:variable name="table-summary">
      <xsl:call-template name="dbhtml-attribute">
        <xsl:with-param name="pis"
                        select="processing-instruction('dbhtml')"/>
        <xsl:with-param name="attribute" select="'table-summary'"/>
      </xsl:call-template>
    </xsl:variable>

    <xsl:variable name="cellpadding">
      <xsl:call-template name="dbhtml-attribute">
        <xsl:with-param name="pis"
                        select="processing-instruction('dbhtml')"/>
        <xsl:with-param name="attribute" select="'cellpadding'"/>
      </xsl:call-template>
    </xsl:variable>

    <xsl:variable name="cellspacing">
      <xsl:call-template name="dbhtml-attribute">
        <xsl:with-param name="pis"
                        select="processing-instruction('dbhtml')"/>
        <xsl:with-param name="attribute" select="'cellspacing'"/>
      </xsl:call-template>
    </xsl:variable>

    <xsl:variable name="toc">
      <xsl:call-template name="dbhtml-attribute">
        <xsl:with-param name="pis"
                        select="processing-instruction('dbhtml')"/>
        <xsl:with-param name="attribute" select="'toc'"/>
      </xsl:call-template>
    </xsl:variable>

    <xsl:variable name="toc.params">
      <xsl:call-template name="find.path.params">
        <xsl:with-param name="table" select="normalize-space($generate.toc)"/>
      </xsl:call-template>
    </xsl:variable>

    <div class="{name(.)}">
      <xsl:apply-templates select="$title"/>
      <xsl:if test="contains($toc.params, 'toc') and $toc != '0'">
        <xsl:call-template name="process.qanda.toc"/>
      </xsl:if>
      <xsl:apply-templates select="$preamble"/>
      <table summary="Q and A Set">
        <xsl:if test="$table-summary != ''">
          <xsl:attribute name="summary">
            <xsl:value-of select="$table-summary"/>
          </xsl:attribute>
        </xsl:if>

        <xsl:if test="$cellpadding != ''">
          <xsl:attribute name="cellpadding">
            <xsl:value-of select="$cellpadding"/>
          </xsl:attribute>
        </xsl:if>

        <xsl:if test="$cellspacing != ''">
          <xsl:attribute name="cellspacing">
            <xsl:value-of select="$cellspacing"/>
          </xsl:attribute>
        </xsl:if>

        <col align="left">
          <xsl:attribute name="width">
            <xsl:choose>
              <xsl:when test="$label-width != ''">
                <xsl:value-of select="$label-width"/>
              </xsl:when>
              <xsl:otherwise>1%</xsl:otherwise>
            </xsl:choose>
          </xsl:attribute>
        </col>
        <tbody>
          <xsl:apply-templates select="qandaentry|qandadiv"/>
        </tbody>
      </table>
    </div>
  </xsl:template>

  <xsl:template name="process.qanda.toc">
    <dl>
      <xsl:apply-templates select="qandadiv" mode="qandatoc.mode"/>
      <xsl:apply-templates select="qandaentry" mode="qandatoc.mode"/>
    </dl>
    <hr />
  </xsl:template>

  <xsl:template name="deflabel">
    <xsl:choose>
      <xsl:when test="ancestor-or-self::*[@defaultlabel]">
        <xsl:value-of select="(ancestor-or-self::*[@defaultlabel])[last()]
                              /@defaultlabel"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:value-of select="qanda.defaultlabel"/>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template name="label.punct">
    <xsl:variable name="deflabel">
      <xsl:call-template name="deflabel"/>
    </xsl:variable>
    <xsl:if test="$deflabel = 'number' or $deflabel = ''">
      <xsl:text>) </xsl:text>
    </xsl:if>
  </xsl:template>

  <xsl:template name="td.nbsp">
    <xsl:variable name="deflabel">
      <xsl:call-template name="deflabel"/>
    </xsl:variable>
    <xsl:if test="$deflabel != 'none' and $deflabel != ''">
      <td>&#x00A0;</td>
    </xsl:if>
  </xsl:template>

  <xsl:template match="qandaentry/revhistory/revision">
    <xsl:variable name="revnumber" select=".//revnumber"/>
    <xsl:variable name="revdate"   select=".//date"/>
    <xsl:variable name="revauthor" select=".//authorinitials"/>
    <xsl:variable name="revremark" select=".//revremark|.//revdescription"/>
    <tr class="{name(.)}">
      <xsl:call-template name="td.nbsp"/>
      <td align="right">
        <p class="faqrev">
          <xsl:call-template name="gentext">
            <xsl:with-param name="key" select="'Revision'"/>
          </xsl:call-template>
          <xsl:call-template name="gentext.space"/>
          <xsl:apply-templates select="$revnumber"/>
          <xsl:text> on </xsl:text>
          <xsl:apply-templates select="$revdate"/>
          <xsl:choose>
            <xsl:when test="count($revauthor)=0">
              <xsl:call-template name="dingbat">
                <xsl:with-param name="dingbat">nbsp</xsl:with-param>
              </xsl:call-template>
            </xsl:when>
            <xsl:otherwise>
              <xsl:text> by </xsl:text>
              <xsl:apply-templates select="$revauthor"/>
            </xsl:otherwise>
          </xsl:choose>
<!--
        <xsl:if test="$revremark">
          <br /><xsl:apply-templates select="$revremark"/>
        </xsl:if> 
-->
        </p>
      </td>
    </tr>
  </xsl:template>

  <xsl:template match="qandaentry">
    <xsl:apply-templates select="question"/>
    <tr>
      <xsl:call-template name="td.nbsp"/>
      <td><br /></td>
    </tr>
    <xsl:apply-templates select="answer"/>
    <xsl:apply-templates select="revhistory/revision"/>
    <tr>
      <xsl:call-template name="td.nbsp"/>
      <td><hr /></td>
    </tr>
  </xsl:template>

  <xsl:template match="question">
    <xsl:variable name="deflabel">
      <xsl:call-template name="deflabel"/>
    </xsl:variable>

    <tr class="{name(.)}">
      <xsl:variable name="id">
        <xsl:call-template name="object.id">
          <xsl:with-param name="object" select=".."/>
        </xsl:call-template>
      </xsl:variable>
      <xsl:if test="$deflabel != 'none' and $deflabel != ''">
        <td id="{$id}" align="left" valign="top">
          <b>
            <xsl:apply-templates select="." mode="label.markup"/>
            <xsl:call-template name="label.punct"/>
          </b>
        </td>
      </xsl:if>
      <td align="left" valign="top">
        <xsl:apply-templates select="*[name(.) != 'label']"/>
      </td>
    </tr>
  </xsl:template>

  <xsl:template name="answer.qanda">
    <tr class="{name(.)}">
      <td align="left" valign="top">
        <b>
          <xsl:variable name="answer.label">
            <xsl:apply-templates select="." mode="label.markup"/>
          </xsl:variable>
          <xsl:copy-of select="$answer.label"/>
        </b>
      </td>
      <td align="left" valign="top">
        <xsl:apply-templates select="*[name(.) != 'label']"/>
      </td>
    </tr>
  </xsl:template>

  <xsl:template name="answer.number">
    <tr class="{name(.)}">
      <xsl:call-template name="td.nbsp"/>
      <td align="left" valign="top">
        <span class="faq_a_leadin">Answer:</span>
      </td>
    </tr>
    <tr class="{name(.)}">
      <xsl:call-template name="td.nbsp"/>
      <td align="left" valign="top">
        <xsl:apply-templates select="*[name(.) != 'label']"/>
      </td>
    </tr>
  </xsl:template>

  <xsl:template name="answer.none">
    <tr class="{name(.)}">
      <td align="left" valign="top">
        <xsl:apply-templates select="*[name(.) != 'label']"/>
      </td>
    </tr>
  </xsl:template>

  <xsl:template name="answer.default">
    <xsl:param name="label"/>
    <xsl:choose>
      <xsl:when test="$label = ''">
        <xsl:call-template name="answer.none"/>
      </xsl:when>
      <xsl:otherwise>
        <tr class="{name(.)}">
          <td align="left" valign="top">
            <b>
              <xsl:value-of select="$label"/>
            </b>
          </td>
          <td align="left" valign="top">
            <xsl:apply-templates select="*[name(.) != 'label']"/>
          </td>
        </tr>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="answer">
    <xsl:variable name="deflabel">
      <xsl:call-template name="deflabel"/>
    </xsl:variable>
    <xsl:choose>
      <xsl:when test="$deflabel = 'qanda'">
        <xsl:call-template name="answer.qanda"/>
      </xsl:when>
      <xsl:when test="$deflabel = 'number'">
        <xsl:call-template name="answer.number"/>
      </xsl:when>
      <xsl:when test="$deflabel = 'none' or $deflabel = ''">
        <xsl:call-template name="answer.none"/>
      </xsl:when>
      <xsl:otherwise>
        <xsl:call-template name="answer.default">
          <xsl:with-param name="label" select="$deflabel"/>
        </xsl:call-template>
      </xsl:otherwise>
    </xsl:choose>
  </xsl:template>

  <xsl:template match="question" mode="qandatoc.mode">
    <xsl:variable name="firstch" select="(*[name(.)!='label'])[1]"/>
    <dt><p>
      <xsl:apply-templates select="." mode="label.markup"/>
      <xsl:call-template name="label.punct"/>
      <a>
        <xsl:attribute name="href">
          <xsl:call-template name="href.target">
            <xsl:with-param name="object" select=".."/>
          </xsl:call-template>
        </xsl:attribute>
        <xsl:value-of select="$firstch"/>
      </a>
    </p></dt>
  </xsl:template>

</xsl:stylesheet>
