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

<xsl:stylesheet version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform">

<xsl:template name="top_banner">

<script type="text/javascript">
var treetop="<xsl:value-of select="$treetop"/>";
document.write('&lt;script type="text/javascript" src="',treetop,'doc/Utilities/HTML/banner_top.js"&gt;&lt;\/script&gt;');
</script>

</xsl:template>

</xsl:stylesheet>
