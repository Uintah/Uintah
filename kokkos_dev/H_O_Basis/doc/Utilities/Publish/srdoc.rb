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


# Classes to make it easier (har har!) to embed ruby code in
# documents.  Still rather primitive.  Add code as needed.  See
# doc/User/FAQ/faq.rxml for an example of usage.


module Constants
  SCI_SoftwareURL = "http://software.sci.utah.edu/"
end

### Source type mixins:

module Text_Source
  def commentBeg()
    "#"
  end

  def commentEnd()
    ""
  end
end

module TeX_Source
  def commentBeg
    "%"
  end
  def commentEnd()
    ""
  end
end

module XML_Source
  def commentBeg
    "<!--"
  end

  def commentEnd
    "-->"
  end

  def sysEntityP(entity, ext)
    a = entity.split(".")
    case a.size
    when 1
      dir = "."
      file = a[0]
    when 2
      dir = a[0]
      file = a[1]
    else
      raise "Bogus entity"
    end
    "<!ENTITY #{entity} SYSTEM \"#{dir}/#{file}.#{ext}\">"
  end

  def docType(rootElement, dtd, entities=nil, sysentities=nil)
    docType = "<!DOCTYPE #{rootElement} SYSTEM \"#{dtd}\""
    if entities or sysentities
      docType += " [\n"
    end
    if entities != nil
      entities.each do |key, value|
	docType += "<!ENTITY #{key} \"#{value}\">\n"
      end
    end
    if sysentities != nil
      sysentities.each do |name|
	docType += sysEntity(name) + "\n"
      end
    end
    if entities or sysentities
      docType += "]"
    end
    docType += ">\n"
  end
end

module DocBook_Source
  include(XML_Source)
  def insertDocType(rootElement, entities=nil, sysentities=nil)
    dtd = ENV["DB_DTD"]
    print(docType(rootElement, dtd, entities, sysentities))
  end
end

### Output mode mixins:

module Print_Output
  include(Constants)

  # Convert tree absolute url to a sci website link.
  def treeUrl(url)
    raise "Bogus url [#{url}]" if not url =~ /^(doc|src)\//
    SCI_SoftwareURL + url
  end

end

module HTML_Output
  # Convert tree absolute url to doc relative url.
  def treeUrl(url)
    raise "Bogus url [#{url}]" if not url =~ /^(doc|src)\//
    Doc.docRoot() + url
  end
end

module XML_Print_Output
  include(Print_Output)

  # Return a system entity which refers to a xml file which is
  #  "optimized" for translation to print output.
  def sysEntity(entity)
    sysEntityP(entity, "pxml")
  end
end

module XML_HTML_Output
  include(HTML_Output)

  # Return a system entity which refers to a xml file which is
  #  "optimized" for translation to html.
  def sysEntity(entity)
    sysEntityP(entity, "hxml")
  end
end

### Document classes

# Base class for instantiable doc classes.  Includes a "factory"
# method ("create") for creating an object appropriate to the
# document's output mode.  See, for instance, doc/User/FAQ/faq.rxml
# for sample usage.

class Doc
  
  # Document types
  DocBook = 0
  HTML = 1
  TeX = 2
  Text = 3

  Edition = "edition.xml"

  # Create a document object.
  def Doc.create(docType, nem=true)
    outputMode = getOutputMode()
    doc = case docType
	  when DocBook
	    case outputMode
	    when "html"
	      DocBook_HTML.new()
	    when "print"
	      DocBook_Print.new()
	    when nil
	      raise("OUTPUT_MODE is undefined!")
	    end
	  when TeX
	    case outputMode
	    when "html"
	      TeX_HTML.new()
	    when "print"
	      TeX_Print.new()
	    when nil
	      TeXDoc.new()
	    end
	  when HTML
	    HTMLDoc.new()
	  when Text
	    TextDoc.new()
	  else
	    raise "Unsupported doc type"
	  end
    @@edition = getEdition()
    doc.insertNoEditMeMsg() if nem == true
    doc
  end

  # Figure out if we be doin' print or html output.
  def Doc.getOutputMode()
    modeString = ENV["OUTPUT_MODE"]
    raise "Invalid output mode" if !(modeString == "print" || modeString == "html" || modeString == nil)
    modeString
  end

  # Execute the given block in the context of the given directory.
  def Doc.inDir(dir)
    pwd = Dir.pwd
    Dir.chdir(dir)
    r = yield(pwd)
    Dir.chdir(pwd)
    r
  end

  def Doc.docRootP()
    raise "Can't find root of doc tree" if Dir.pwd == "/"
    if (FileTest.directory?("doc") && FileTest.directory?("src"))
      ""
    else
      Dir.chdir("..")
      "../" + docRootP()
    end
  end

  def Doc.docRoot()
    Doc.inDir(Dir.pwd) do
      docRootP()
    end
  end

  # Extract the content of the edition.xml file.
  def Doc.getEdition()
    editionFile = nil
    inDir(".") do
      while !FileTest.exists?(Edition)
	raise "Can't find edition.xml" if Dir.pwd == "/"
	Dir.chdir("..")
      end
      editionFile = Dir.pwd + "/" + Edition
    end
    File.open(editionFile) do |f|
      /<edition>(.*)<\/edition>/.match(f.read())[1]
    end
  end

  def Doc.edition()
    @@edition
  end

  # Insert "no edit me" message into current document.
  def insertNoEditMeMsg()
    msg = "* DON'T EDIT ME!  I'm generated.  Edit my source file instead *"
    putComment("*"*msg.size)
    putComment("*" + (" "*(msg.size-2)) + "*")
    putComment(msg)
    putComment("*" + (" "*(msg.size-2)) + "*")
    putComment("*"*msg.size)
  end

  # Return value of <edition> element from edition.xml.
  def edition()
    Doc.edition
  end

  # Generate a comment with the given text.
  def comment(text)
    commentBeg + " " + text + " " + commentEnd
  end

  # Insert command into current document.
  def putComment(text)
    print(comment(text) + "\n")
  end

  # Is we a doin' html output?
  def htmlOutput?()
    kind_of?(HTML_Output)
  end
  
  # Execute the given block if we be doin' html output.
  def doIfHTML()
    if htmlOutput?()
      yield()
    else
      ""
    end
  end

  # Is we a doin' print output?
  def printOutput?()
    kind_of?(Print_Output)
  end

  # Execute the given block if we be doin' print output.
  def doIfPrint()
    if printOutput?()
      yield()
    else
      ""
    end
  end

  # Include a file with eruby substitution.
  def include(fileName)
    file = File.open(fileName)
    compiler = ERuby::Compiler.new
    c = compiler.compile_file(file)
    eval(c,TOPLEVEL_BINDING)
  end

end

# Instantiable document classes.  Each is derived from class Doc and
# then mixes in a source type module and perhaps an output type
# module.  Objects of these types are instantiated by class Doc's
# create function (don't make much sense to do it otherwise).

# "Plain" ol' text document, e.g. Doxygen config file.
class TextDoc < Doc
  include(Text_Source)
end

# HTML doc, i.e. all index.rhtml files.
class HTMLDoc < Doc
  include(XML_Source)
end

# TeX document - no knowledge of output mode.
class TeXDoc < Doc
  include(TeX_Source)
end

# TeX document generating HTML output.
class TeX_HTML < Doc
  include(TeX_Source)
  include(HTML_Output)
end

# TeX document generating print output.
class TeX_Print < Doc
  include(TeX_Source)
  include(Print_Output)
end

# DocBook document generating HTML output.
class DocBook_HTML < Doc
  include(DocBook_Source)
  include(XML_HTML_Output)
end

# DocBook document generating print output.
class DocBook_Print < Doc
  include(DocBook_Source)
  include(XML_Print_Output)
end

