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
# Build an index.html file for all html files in the specified package.
#
# This script is used by doc/Developers/Modules/Makefile to build
# an index.html file for each package.
#

class Entry
  attr_reader :base, :file
  def initialize(file)
    @file = file
    @base = File.basename(file)
  end
  def <=>(e)
    @base <=> e.base
  end
  def get
    "<td><a href='#{@file}'>#{File.basename(@file, '.html')}</a></td>"
  end
end

class LatexEntry < Entry
  @@count = 0
  def initialize(file)
    super
    @@count += 1
  end
  def get
    "<td><a href='#{@file}'>#{File.basename(@file, '.html')}</a><sup>*</sup></td>"
  end
  def LatexEntry.count
    @@count
  end
end

def main

  packageName = ARGV[0]
  xmlDir = ARGV[1]
  texDir = ARGV[2]

  begin
    File.open("#{packageName}.html", "w") do |index|

      # Insert boilerplate at top of file
      index.print <<EndOfString
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
EndOfString

      # Generate page heading.
      index.print("<p class='title'> #{packageName} Module Descriptions</p>\n")
      index.print("<hr size='1'/>")

      # Generate index of modules.  Alphabetical order in 3 columns.
      entries = []
      Dir["#{xmlDir}/*.html"].each do |f|
          entries << Entry.new(f)
      end
		    
      Dir["#{texDir}/*/*/*.html"].each do |f|
        entries << LatexEntry.new(f)
      end
      entries.sort!
      numCols = 3
      numRows = entries.size() / numCols
      extras = entries.size() % numCols
      numRowsArray = [ numRows, numRows, numRows ]
      if extras > 0
        numRowsArray[0] += 1
      end
      if extras > 1
        numRowsArray[1] += 1
      end
      etable = []
      etable[0] = entries[0, numRowsArray[0]]
      etable[1] = entries[numRowsArray[0], numRowsArray[1]]
      etable[2] = entries[numRowsArray[0] + numRowsArray[1], numRowsArray[2]]
      index.print("<table align='center' cellspacing='5'>\n")
      numRows.times do |i|
        index.print("<tr>\n")
        numCols.times do |j|
          index.print((etable[j][i]).get, "\n")
        end
        index.print("</tr>\n")
      end
      if extras > 0
        index.print("<tr>\n")
        index.print((etable[0][numRows]).get, "\n")
        if extras > 1
          index.print((etable[1][numRows]).get, "\n")
        end        
        index.print("</tr>\n")
      end
      index.print("</table>\n")

      if LatexEntry.count > 0
        index.print("<p><sup>*</sup>Denotes additional documentation for a module.</p>\n")
      end

      # Generate end of file boilerplate.
      index.print <<EndOfString
<script language="JavaScript">
document.write('<script language="JavaScript" src="',treetop,'doc/Utilities/HTML/banner_bottom.js"><\\/script>');
</script>
</body>
</html>
EndOfString

      end

  rescue => oops
    print("Unable to build #{packageName}.html for package\n")
    print("Reason: ", oops.message, "\n")
  end

end

main
