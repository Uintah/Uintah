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

catch {rename CardioWave_CreateModel_ReclassifyInteriorTets ""}

itcl_class CardioWave_CreateModel_ReclassifyInteriorTets {
    inherit Module

    constructor {config} {
	set name ReclassiyInteriorTets
	set_defaults
    }

    method set_defaults {} {	
        global $this-threshold
        global $this-tag
        set $this-threshold 0.01
        set $this-tag 1
    }

    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
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
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes

        global $this-threshold
	make_entry $w.f.e "Distance threshold:" $this-threshold \
		"$this-c needexecute"
	make_entry $w.f.c "New classification tag:" $this-tag \
		"$this-c needexecute"
	pack $w.f.e $w.f.c -side top -fill x -expand 1
   }
}