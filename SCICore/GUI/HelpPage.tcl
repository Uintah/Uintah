#
#  HelpPage.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#

#
# DON'T SOURCE THIS FILE...it's taken care of in NetworkEditor.tcl.
#

#
# Use `helpPage path' where path is the FULL path of the html file.
#

#
# Use $sci_root/help/boilerplate.html as a guide for making a help page.
# Be sure the title starts with `SCIRun Help' or things break!!
#

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
    # "(Netscape: SCIRun Help"
    set result [regexp {(0x[0-9a-fA-F]+)\ *\(Netscape:\ SCIRun\ Help} $wids line hex]
    set sci_netscape_wid $hex
}


# Initialize our netscape pid and wid variables.

global sci_netscape_pid sci_netscape_wid
set sci_netscape_pid ""
set sci_netscape_wid ""

