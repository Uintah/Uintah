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

