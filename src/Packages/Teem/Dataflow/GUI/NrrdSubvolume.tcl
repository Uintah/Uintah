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
 #  NrrdSubvolume.tcl: The NrrdSubvolume UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Filters_NrrdSubvolume ""}

itcl_class Teem_Filters_NrrdSubvolume {
    inherit Module
    constructor {config} {
        set name NrrdSubvolume
        set_defaults
    }
    method set_defaults {} {
        global $this-minAxis0
        global $this-maxAxis0
        global $this-minAxis1
        global $this-maxAxis1
        global $this-minAxis2
        global $this-maxAxis2
        set $this-minAxis0 0
        set $this-maxAxis0 127
        set $this-minAxis1 0
        set $this-maxAxis1 127
        set $this-minAxis2 0
        set $this-maxAxis2 127
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v -width 6
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
        wm minsize $w 150 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	make_entry $w.f.minAxis0 "Min: Axis 0" $this-minAxis0 \
		"$this-c needexecute"
	make_entry $w.f.maxAxis0 "Max: Axis 0" $this-maxAxis0 \
		"$this-c needexecute"
	make_entry $w.f.minAxis1 "Min: Axis 1" $this-minAxis1 \
		"$this-c needexecute"
	make_entry $w.f.maxAxis1 "Max: Axis 1" $this-maxAxis1 \
		"$this-c needexecute"
	make_entry $w.f.minAxis2 "Min: Axis 2" $this-minAxis2 \
		"$this-c needexecute"
	make_entry $w.f.maxAxis2 "Max: Axis 2" $this-maxAxis2 \
		"$this-c needexecute"
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.minAxis0 $w.f.maxAxis0 $w.f.minAxis1 $w.f.maxAxis1 \
		$w.f.minAxis2 $w.f.maxAxis2 $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
