#
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

#
# Build an index.html file for all html files in the current directory.
#
# This script is used by doc/Developers/Modules/Makefile to build
# an index.html file for each package description directory
# in the Modules directory.
#

packageName = ARGV[0]
packageDir = ARGV[1]

begin
  File.open("#{packageName}.html", "w") do |index|

# Insert boilerplate at top of file
  index.print <<ZZZZZ
<!doctype html public "-//w3c//dtd html 4.0 transitional//en">
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
<html>
<head>
<script language="JavaScript">
var treetop="";
var path = location.pathname;
while (path.substr(path.lastIndexOf("/")+1) != "doc") {
treetop += "../";
path = path.substr(0, path.lastIndexOf("/"));
}
document.write("<link href='",treetop,"doc/Utilities/HTML/doc_styles.css' rel='stylesheet' type='text/css'>")
</script>
<title></title>
</head>
<body>
<script language="JavaScript">
document.write('<script language="JavaScript" src="',treetop,'doc/Utilities/HTML/banner_top.js"><\\/script>');
</script>
ZZZZZ

  # Generate page heading.
#  package = File.basename(`pwd`).strip
  index.print("<p class='title'> #{packageName} Module Descriptions</p>\n")
  index.print("<hr size='1'/>")

  # Generate index of modules.  Alphabetical order in 3 columns.
  files = Dir["#{packageDir}/*.html"].sort
  numCols = 3
  numRows = files.size() / numCols
  extras = files.size() % numCols
  numRowsArray = [ numRows, numRows, numRows ]
  if extras > 0
    numRowsArray[0] += 1
  end
  if extras > 1
    numRowsArray[1] += 1
  end
  table = []
  table[0] = files[0, numRowsArray[0]]
  table[1] = files[numRowsArray[0], numRowsArray[1]]
  table[2] = files[numRowsArray[0] + numRowsArray[1], numRowsArray[2]]
  index.print("<table align='center' cellspacing='5'>\n")
  numRows.times do |i|
    index.print("<tr>\n")
    numCols.times do |j|
      file = table[j][i]
      index.print("<td><a href='#{file}'>#{File.basename(file, '.html')}</a></td>\n")
    end
    index.print("</tr>\n")
  end
  if extras > 0
    index.print("<tr>\n")
    file = table[0][numRows]
    index.print("<td><a href='#{file}'>#{File.basename(file, '.html')}</a></td>\n")
    if extras > 1
      file = table[1][numRows]
      index.print("<td><a href='#{file}'>#{File.basename(file, '.html')}</a></td>\n")
    end        
    index.print("</tr>\n")
  end
  index.print("</table>\n")

  # Generate end of file boilerplate.
  index.print <<ZZZZZ
<script language="JavaScript">
document.write('<script language="JavaScript" src="',treetop,'doc/Utilities/HTML/banner_bottom.js"><\\/script>');
</script>
</body>
</html>
ZZZZZ

  end

rescue
  print("Unable to build #{packageName}.html for package\n")
  print($!, "\n")
end
