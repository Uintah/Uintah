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

##
 #  LocateNbrhd.tcl: General Mesh interpolation module
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #  Copyright (C) 1997 SCI Group
 #  Log Information:
 ##

catch {rename LocateNbrhd ""}

itcl_class SCIRun_Fields_LocateNbrhd {
    inherit Module
    constructor {config} {
        set name LocateNbrhd
        set_defaults
    }
    method set_defaults {} {
	global $this-method
	set $this-method project
	global $this-zeroTCL
	set $this-zeroTCL 0
	global $this-potMatTCL
	set $this-potMatTCL 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
	
        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-method
	make_labeled_radio $w.f.method "Method: " "" \
		top $this-method \
		{{"S2->S1 Project" project}}
	global $this-zeroTCL
	checkbutton $w.f.zero -text "Don't use mesh node zero" -variable $this-zeroTCL
	global $this-potMatTCL
	checkbutton $w.f.pot -text "Build potential difference matrix (ground=0)" -variable $this-potMatTCL
	pack $w.f.method $w.f.zero $w.f.pot -side top -fill x
        pack $w.f -side top -expand yes
    }
}
