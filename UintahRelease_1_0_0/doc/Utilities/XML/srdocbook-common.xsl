<?xml version="1.0" encoding="utf-8"?>

<!--
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
-->

<!-- The following templates override those found in docbook.xsl and
     docbook-chunk.xsl --> 

<xsl:stylesheet xmlns:xsl="http://www.w3.org/1999/XSL/Transform" version="1.0">

  <!-- Include xref gen text customizations. -->
  <xsl:param name="local.l10n.xml" select="document('srgentext.xml')"/>

  <!-- treetop is relative path to top of sr tree -->
  <xsl:param name="treetop"/>

  <!-- Include customizations of qandaset -->
  <xsl:include href="srqandaset.xsl"/>

  <!-- Load javascript tools -->
  <xsl:template name="user.head.content">
    <script type="text/javascript">
      <xsl:attribute name="src">
	<xsl:value-of select="$treetop"/>/doc/Utilities/HTML/tools.js<xsl:text/>
      </xsl:attribute>
    </script>
    <script type="text/javascript">var doc = new DocBookDocument();</script>
  </xsl:template>

  <!-- Generate page "pre-content" -->
  <xsl:template name="user.header.content">
    <script type="text/javascript">doc.preContent();</script>
  </xsl:template>

  <!-- Generate page "post-content" -->
  <xsl:template name="user.footer.content">
    <script type="text/javascript">doc.postContent(); </script>
  </xsl:template>

  <!-- Change type from a 'charseq' to a 'monoseq' -->
  <xsl:template match="type">
    <xsl:call-template name="inline.monoseq"/>
  </xsl:template>

</xsl:stylesheet>
