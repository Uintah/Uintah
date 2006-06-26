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


#  HelpPage.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#  Copyright (C) 1995 SCI Group

# DON'T SOURCE THIS FILE...it's taken care of in NetworkEditor.tcl.

# Use `helpPage path' where path is the FULL path of the html file.

# Use $sci_root/help/boilerplate.html as a guide for making a help page.
# Be sure the title starts with `Dataflow Help' or things break!!

proc helpPage {path} {
    global sci_netscape_pid sci_netscape_wid

    if {$sci_netscape_pid == ""} {
	newPage $path
    } else {
	set output [exec ps]
	if {[string first $sci_netscape_pid $output] == -1} {
	    newPage $path
	} else {
	    exec netscape -id $sci_netscape_wid -remote openFile($path)
	}
    }
}

# newPage is used by helpPage
proc newPage {path} {
    global sci_netscape_pid sci_netscape_wid

    set sci_netscape_pid [exec netscape $path &]
    # We have to wait for the new window to appear on the wid list...
    # Don't make any less.
    after 4000
    set wids [exec xlswins]
    # Search for the hex wid followed by the correct help page name prefix:
    # "(Netscape: Dataflow Help"
    set result [regexp {(0x[0-9a-fA-F]+)\ *\(Netscape:\ Dataflow\ Help} $wids line hex]
    set sci_netscape_wid $hex
}


# Initialize our netscape pid and wid variables.

global sci_netscape_pid sci_netscape_wid
set sci_netscape_pid ""
set sci_netscape_wid ""

