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

require "../src/Core/CCA/tools/strauss/ruby/sr_util"
require "erb/erb"
require "rexml/document"
include REXML

args = getArgs
for i in 0...args.length
  print "YOYOYOYOYO'"+args[i]+"'\n"
end

doc = Document.new File.new( args[0] )
package = doc.root.attributes["name"]
port = doc.root.elements["n:port"]
portname = port.attributes["name"]
method = port.elements["n:method"]
methodname = method.attributes["name"]
methodret = method.attributes["retType"]

def outDefArgs( methodname , doc )
  m = doc.root.elements["n:port/n:method[@name='"+methodname+"']"]
  out = ""
  if (m) then m.elements.each("n:argument") { |arg| out += arg.attributes["type"]+" "+arg.attributes["name"]+"," } end 
  out.slice!(out.length-1,out.length)
  print out
end

#same as outDefArgs but starts with a comma is args exist
def commaoutDefArgs( methodname , doc )
  m = doc.root.elements["n:port/n:method[@name='"+methodname+"']"]
  out = ""
  if (m) then m.elements.each("n:argument") { |arg| out += ","+arg.attributes["type"]+" "+arg.attributes["name"] } end
  print out
end


def outCallArgs( methodname , doc )
  m = doc.root.elements["n:port/n:method[@name='"+methodname+"']"]
  out = ""
  if (m) then m.elements.each("n:argument") { |arg| out += arg.attributes["name"]+"," } end
  out.slice!(out.length-1,out.length)
  print out
end

makename = args[2]
bridgeC = args[3]

File.open( args[1] ) { |fh|
  erb = ERb.new( fh.read )
  ret2sr erb.result( binding )
}

