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
<script type="text/javascript">
var treetop="";
var path = location.pathname;
if (path.charAt(path.length-1) == "/") {
  path += "bogus.html"
}
var base = path.substr(path.lastIndexOf("/")+1);
var roottag = "doc";
while (base != roottag && base != "") {
  treetop += "../";
  path = path.substr(0, path.lastIndexOf("/"));
  base = path.substr(path.lastIndexOf("/")+1);
}
var inDocTree = base == roottag;
if (inDocTree) {
  document.write("<link href='",treetop,"doc/Utilities/HTML/moduleindex.css' rel='stylesheet' type='text/css'/>")
}
</script>
<title>#{packageName} Module Descriptions</title>
</head>
<body>
<script type="text/javascript">
if (inDocTree) {
  document.write('<script type="text/javascript" src="',treetop,'doc/Utilities/HTML/banner_top.js"><\\/script>');
}
</script>
EndOfString

      version = File.open("../../edition.xml") do |f|
        /<edition>(.*)<\/edition>/.match(f.read())[1]
      end

      index.print("<br/>")

      # Generate index of modules.  Alphabetical order in 3 columns.
      entries = []
      Dir["#{xmlDir}/*.html"].each do |f|
          entries << Entry.new(f)
      end
		    
      Dir["#{texDir}/*/*/*.html"].each do |f|
        entries << LatexEntry.new(f)
      end
      entries.sort!
      index.print("<div class=\"block-center\">\n")
      index.print("<a href=\"index.html\">Back to Package Listing</a>\n")
      index.print("<table class=\"default\" cellspacing='0' border='1' l
cellpadding='2'> \n")
      # Generate page title and subtitle
      etable = []
      if entries.size() <= 25
	numCols = 1
	numRows = entries.size()
	numRowsArray = [ numRows ]
	etable[0] = entries
	extras = 0
      elsif entries.size() <= 50
	numCols = 2
	numRows = entries.size() / numCols
	extras = entries.size() % numCols
	numRowsArray = [ numRows, numRows ]
	extras.times do |i|
	  numRowsArray[i] += 1
	end
	etable[0] = entries[0, numRowsArray[0]]
	etable[1] = entries[numRowsArray[0], numRowsArray[1]]
      else
	numCols = 3
	numRows = entries.size() / numCols
	extras = entries.size() % numCols
	numRowsArray = [ numRows, numRows, numRows ]
	extras.times do |i|
	  numRowsArray[i] += 1
	end
	etable = []
	etable[0] = entries[0, numRowsArray[0]]
	etable[1] = entries[numRowsArray[0], numRowsArray[1]]
	etable[2] = entries[numRowsArray[0] + numRowsArray[1], numRowsArray[2]]
      end

      index.print("
<th colspan=\"#{numCols}\">
<span class=\"title\">#{packageName} Module Descriptions</span>
<br/>
<span class=\"subtitle\">alphabetically [<a class=\"alt\" href=\"#{packageName}_bycat.html\">by category</a>]</span>
<br/>
<span class=\"subtitle\">for SCIRun version #{version}</span>
</th>
\n")
      numRows.times do |i|
        index.print("<tr>\n")
        numCols.times do |j|
          index.print((etable[j][i]).get, "\n")
        end
        index.print("</tr>\n")
      end
      extras.times do |i|
	index.print("<tr>\n")
	index.print((etable[i][numRows]).get, "\n")
	index.print("</tr>\n")
      end

      index.print("</table>\n")

      if LatexEntry.count > 0
        index.print("<p><sup>*</sup>Denotes additional documentation for a module.</p>\n")
      end

      index.print("</div>\n")

      # Generate end of file boilerplate.
      index.print <<EndOfString
<script type="text/javascript">
if (inDocTree) {
  document.write('<script type="text/javascript" src="',treetop,'doc/Utilities/HTML/banner_bottom.js"><\\/script>');
}
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
