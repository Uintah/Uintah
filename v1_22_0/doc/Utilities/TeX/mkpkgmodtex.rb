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
# Read list of tex files on stdin, sort them using file basename only, and then
# write sorted list to stdout enclosing each filename in a tex \input
# statement.  This tool is used by doc/Developer/Modules/Makefile to generate a
# sorted list of module description tex files generated from xml and hand
# crafted tex module description files.
#

class Entry
  attr_reader :file, :base
  def initialize(file)
    @file = file
    @base = File.basename(file)
  end
  def <=>(e)
    @base <=> e.base
  end
end

entries = []
$stdin.each do |e|
  entries << Entry.new(e.chomp())
end

entries.sort.each do |e|
  dir = File.dirname(e.file)
  if dir =~ /TeX/
    $stdout.print("\\renewcommand{\\ModuleRefFigName}[1]{#{dir}/#1}\n")
  end
  $stdout.print("\\input{#{e.file}}
\\clearpage
\\pagebreak\n")
end
