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
 #  SelectElements.tcl: The ChangeCellType UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename SCIRun_Fields_SelectElements ""}

itcl_class SCIRun_Fields_SelectElements {
    inherit Module
    constructor {config} {
        set name SelectElements
        set_defaults
    }
    method set_defaults {} {
        global $this-keep-all-nodes
        set $this-keep-all-nodes 0
        global $this-value
        set $this-value 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 300 80
	global $this-keep-all-nodes
	checkbutton $w.b -text "Keep all nodes" -variable $this-keep-all-nodes
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-value
	label $w.f.l -text "Element value" -width 10 -just left
	entry $w.f.e -width 10 -textvariable $this-value
	bind $w.f.e <KeyPress-Return> "$this-c needexecute"
	pack $w.f.l $w.f.e -side left -expand 1 -fill x
	pack $w.b $w.f -expand 1 -fill x
    }
}
