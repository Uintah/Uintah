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

#!/usr/bin/ruby
# -*- ruby -*-

# srvalidate --
#
# This cgi script validates the results of an XML file file selected by the
# user.  It is invoked by the page srvalidate.html. Output is
# formatted by srvalidate.css.  
#

require "cgi"

XERCESPATH = "/usr/local/bin/xerces-2_1_0"
ENV['CLASSPATH'] = "#{XERCESPATH}/xercesSamples.jar:#{XERCESPATH}/xercesImpl.jar:#{XERCESPATH}/xmlParserAPIs.jar"

DTDDIR = "component.dtd"
TMP_FILE_SUFFIX = "_srvalidate"
JAVA = "/usr/bin/java"

# Validate contents of the 'TempFile' object given by 'tfo'. 'tfo' is a
#  Ruby TempFileObject (slightly extended by the CGI class).
# Return results of validation as a string.
def validate(tfo)
  content = tfo.read
  dtre = /<!DOCTYPE\s+.*\s+(SYSTEM|PUBLIC)\s+(".*")?\s*"(.*\.dtd)"/
  dtmd = dtre.match(content)
  raise "Couldn't parse your document type declaration" if dtmd == nil
  dtd = File.basename(dtmd[3])
  case dtd
  when "component.dtd"
    content.sub!(dtre, "<!DOCTYPE component SYSTEM \"#{DTDDIR}/#{dtd}\"")
  end
  fileName = tfo.local_path + TMP_FILE_SUFFIX
  File.open(fileName, "w") { |f|
    f.write(content)
  }
  result = `#{JAVA} dom.Counter -v #{fileName} 2>&1`
  File.delete(fileName)
  result
end

# Generate page from results of validation.
def generateResultReply
  cgi = CGI.new("html4")
  cgi.out {
    cgi.html {
      cgi.head {
	cgi.title { "Validation Results" } +
	  cgi.link({ "rel"=>"stylesheet", "href"=>"http://www.cvrti.utah.edu/~dustman/srvalidate.css"})
      } +
	cgi.body {
	values = cgi['XML_File']
	content = cgi.h3 { "Validation Results for #{values[0].original_filename}" } + cgi.br 
	ra = validate(values[0]).split("\n")
	if ra.size > 1
	  content += cgi.table {
	    s = cgi.tr {
	      cgi.th {"Error"} + cgi.th {"Line"} + cgi.th{"Column"} + cgi.th{"Reason"}
	    }
	    ra.each { |l|
	      if (md = /\[Error\].*:(\d+):(\d+):(.*)/.match(l)) != nil
		s += cgi.tr {
		  cgi.td { "<font class=error>Non-fatal</font>" } + cgi.td { md[1] } + cgi.td { md[2] } + cgi.td { md[3] }
		}
	      elsif (md = /\[Fatal Error\].*:(\d+):(\d+):(.*)/.match(l)) != nil
		s += cgi.tr {
		  cgi.td { "<font class=fatalerror>Fatal</font>" } + cgi.td { md[1] } + cgi.td { md[2] } + cgi.td { md[3] }
		}
	      end
	    }
	    s
	  }
	else
	  content += cgi.p { "Yea! No Errors!" }
	end
	content
      }
    }
  }
end

# Generate an error page if something goes wrong.
def generateErrorReply
  cgi = CGI.new("html4")
  cgi.out {
    cgi.html {
      cgi.head { cgi.title { "Validation Results" } } +
	cgi.body {
	cgi.p { "A really bad thing occurred. Please contact " +
	    cgi.a("mailto://dustman@cvrti.utah.edu") { "Ted Dustman" } + "
with the content of this page." }   +
	  cgi.p { $! } 
      }
    }
  }
end

def main
  begin
    generateResultReply
  rescue
    generateErrorReply
  end
end

main

