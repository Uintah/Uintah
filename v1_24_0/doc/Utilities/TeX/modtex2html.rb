#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


#
# Translate latex version of module description to html
#

require 'fileutils'

def findTreeTop(treetop="")
  if File.basename(Dir.pwd) == "src"
    treetop + ".."
  else
    Dir.chdir("..")
    findTreeTop(treetop + "../")
  end
end

# Return relative path to top of SCIRun tree using simple-minded
# algorithm.  We assume we are searching from the src side of the
# tree.
def treeTop()
  saveDir = Dir.pwd
  treetop = findTreeTop
  Dir.chdir(saveDir)
  treetop
end

# Convert the given tex fragment to a real tex document by inserting
# needed stuff at the top and bottom of the file.  The original file gets
# renamed by appending the string ".O" to its name.
def texFragToTexDoc(infile, outfile)
  File.open(outfile, "w") do |f|
    f.write("\\documentclass[notitlepage]{article}
\\usepackage{graphicx}
\\usepackage{html}
\\usepackage{scirun-doc}
\\input{scirun-doc.tex}
\\newcommand{\\ModuleRef}[3]{\\section{#1} #1 is in Category #2 of Package #3}
\\newcommand{\\Package}[1]{#1}
\\newcommand{\\Category}[1]{#1}
\\newcommand{\\Module}[1]{#1}
\\newcommand{\\ModuleRefSummary}{\\subsection*{Summary}}
\\newcommand{\\ModuleRefUse}{\\subsection*{Use}}
\\newcommand{\\ModuleRefDetails}{\\subsection*{Details}}
\\newcommand{\\ModuleRefNotes}{\\subsection*{Notes}}
\\newcommand{\\ModuleRefCredits}{\\subsection*{Credits}}
\\newcommand{\\ModuleRefSection}[1]{\\subsection*{#1}}
\\newcommand{\\ModuleRefSubSection}[1]{\\subsubsection*{#1}}
\\newcommand{\\ModuleRefSubSubSection}[1]{\\paragraph*{#1}}
%begin{latexonly}
\\newcommand{\\ModuleRefFigName}[1]{#1}
%end{latexonly}
\\begin{htmlonly}
\\newcommand{\\ModuleRefFigName}[1]{../#1}
\\end{htmlonly}
\\begin{document}\n")
    f.write(File.open(infile, "r").readlines)
    f.write("\\end{document}\n")
  end
end

# Convert tex file to html using latex and latex2html tools
def tex2html(moduleName)
  htmlDir = moduleName
  texFile = moduleName + ".tex"
  system("rm -rf #{htmlDir}") if FileTest.exists?(htmlDir)
  latexcmd = "latex -interaction=nonstopmode #{texFile}"
  system("#{latexcmd};#{latexcmd}")
  system("latex2html -split 0 -no_navigation -html_version 4.0 -image_type gif #{texFile}")
  system("rm -f *.dvi *.log *.aux #{htmlDir}/index.html #{htmlDir}/#{moduleName}.css")
end

# Add code to <head> element that figures the top of the tree relative to
#  location of html file. 
def filterHead(c)
  c.sub!("<HEAD>",<<EndOfString
<HEAD>
<script type="text/javascript">
var treetop="";
var path = location.pathname;
if (path.charAt(path.length-1) == "/") {
  path += "bogus.html"
}
var base = path.substr(path.lastIndexOf("/")+1);
var roottag = "src";
while (base != roottag && base != "") {
  treetop += "../";
  path = path.substr(0, path.lastIndexOf("/"));
  base = path.substr(path.lastIndexOf("/")+1);
}
var inDocTree = base == roottag;
if (inDocTree) {
document.write("<link href='",treetop,"doc/Utilities/HTML/srlatex2html.css' rel='stylesheet' type='text/css'/>")
}
</script>
EndOfString
)
end

# Add code to top of <body> element that inserts the standard SCIRun top or
# header banner.
def addTopBanner(c)
  c.sub!("<BODY >", <<EndOfString
<BODY>
<script type="text/javascript">
if (inDocTree) {
  document.write('<script type="text/javascript" src="',treetop,'doc/Utilities/HTML/banner_top.js"><\\/script>');
}
</script>
EndOfString
)
end

# Add code at the bottom  of <body> element that inserts the standard
# SCIRun bottom or footer banner.
def addBottomBanner(c)
  c.sub!("</BODY>", <<EndOfString
<script type="text/javascript">
if (inDocTree) {
  document.write('<script type="text/javascript" src="',treetop,'doc/Utilities/HTML/banner_bottom.js"><\\/script>');
}
</script>
</BODY>
EndOfString
)
end

# Add top and bottom banners to the body element.
def addBanners(c)
  addTopBanner(c)
  addBottomBanner(c)
end

# Remove reference to latex2html generated css file.
def rmCssRef(c, modName)
  c.sub!("<LINK REL=\"STYLESHEET\" HREF=\"#{modName}.css\">", "")
end

# Add standard SR disclaimer.
def addDisclaimer(c)
  disclaimer = <<EndOfString

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

EndOfString
  c.sub!(/$/,disclaimer)
end

# Transform latex2html generated html into SCIRun conformant html.
def filterhtml(moduleName)
  htmlDir = moduleName
  htmlFile = htmlDir + "/" + htmlDir + ".html"
  c = File.open(htmlFile, "r").read
  addDisclaimer(c)
  rmCssRef(c, moduleName)
  filterHead(c)
  addBanners(c)
  File.open(htmlFile, "w").write(c)
end

# main: Coordinate the entire mess.
def main()
  saveDir = Dir.pwd
  texDir = ARGV[0].sub(/\/$/, "")
  Dir.chdir(texDir)
  moduleName = File.basename(texDir)
  texFile = moduleName + ".tex"
  begin
    origTexFile = texFile + ".O"
    FileUtils.copy(texFile, origTexFile)
    texUtils = treeTop() + "/doc/Utilities/TeX"
    ["sty", "tex", "perl"].each do |ext|
      File.symlink("#{texUtils}/scirun-doc.#{ext}", "scirun-doc.#{ext}")
    end
    texFragToTexDoc(origTexFile, texFile)
    tex2html(moduleName)
    filterhtml(moduleName)
  rescue => oops
    print("modtex2html.rb: Something bad happend (#{texFile})\n")
    print("Reason: ", oops.message, "\n")
    print(oops.backtrace.join("\n"))
  ensure
    ["sty", "tex", "perl"].each do |ext|
      symlink = "scirun-doc.#{ext}"
      File.delete(symlink) if FileTest.exists?(symlink)
    end
    if FileTest.exists?(origTexFile)
      File.delete(texFile)
      File.rename(origTexFile, texFile)
    end
    Dir.chdir(saveDir)
  end
end

main
