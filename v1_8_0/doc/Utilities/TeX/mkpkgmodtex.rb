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
# Read list of tex files on stdin, sort them using file basename only, and then
# write sorted list to stdout enclosing each filename in a tex \input
# statement.  This tool is used by doc/User/Guide/Makefile to generate a
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
