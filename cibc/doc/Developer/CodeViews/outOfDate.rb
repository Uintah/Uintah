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

# This program determines if the doxygen generated documents are out
# of date with respect to Doxyconfig.r and .c, .cc, and .h files in the
# src side of the tree. Returns 0 if out of date and 1 if not.

require 'find'
require "../../Utilities/Publish/utils.rb"

begin
  timestamp = File.mtime("html")
  exit(0) if File.mtime("Doxyconfig.r") > timestamp
  sourceMatch = /\.(c|cc|h)$/
  srcRoot = "../../../src"
  dirs = SrcTopLevel.dirs(srcRoot)
  dirs.each do |d|
    Find.find(srcRoot + "/" + d) do |f|
      exit(0) if (f =~ sourceMatch) and (File.mtime(f) > timestamp)
    end
  end
rescue
  exit(0)
end

exit(1)
