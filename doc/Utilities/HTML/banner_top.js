/*
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
*/

// Generate top banner.
document.write('<div class="top-banner-margins">')
document.write('<table border="0" cellspacing="0" cellpadding="0" width="100%" height="91">')
document.write('<tr>')
document.write('<td align="center" width="%100" background="',treetop,'doc/Utilities/Figures/banner_top_fill.jpg">')
document.write("<img src='",treetop,"doc/Utilities/Figures/banner_top.jpg' width='744' height='91' border='0' usemap='#banner'/>")
document.write('</td>')
document.write("<map name='banner'>")
document.write("<area href='http://www.sci.utah.edu' alt='Home' coords='118,37,161,53'/>")
document.write("<area href='http://software.sci.utah.edu' alt='Software' coords='117,62,213,84' />")
document.write("<area href='",treetop, "doc/index.html' alt='Doc Home' coords='223,62,370,84' />")
document.write("<area href='",treetop, "doc/Installation/index.html' alt='Installation' coords='384,62,506,84' />")
document.write("<area href='",treetop, "doc/User/index.html' alt='User' coords='516,62,576,84' />")
document.write("<area href='", treetop, "doc/Developer/index.html' alt='Developer' coords='586,62,690,84' />")
document.write("</map>")
document.write("</tr>")
document.write("</table>")
document.write('</div>')
