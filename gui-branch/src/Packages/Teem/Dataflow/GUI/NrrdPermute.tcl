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
 #  NrrdPermute.tcl: The NrrdPermute UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Filters_NrrdPermute ""}

itcl_class Teem_Filters_NrrdPermute {
    inherit Module
    constructor {config} {
        set name NrrdPermute
        set_defaults
    }
    method set_defaults {} {
        global $this-axis0
        global $this-axis1
        global $this-axis2
        set $this-axis0 0
        set $this-axis1 1
        set $this-axis2 2
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v -width 2
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-axis0
	global $this-axis1
	global $this-axis2
	make_entry $w.f.i "0 -> " $this-axis0 "$this-c needexecute"
	make_entry $w.f.j "1 -> " $this-axis1 "$this-c needexecute"
	make_entry $w.f.k "2 -> " $this-axis2 "$this-c needexecute"
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.i $w.f.j $w.f.k $w.f.b -side top -expand 1 -fill x
    }
}
