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

