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
 #  NrrdCrop.tcl: The NrrdCrop UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Filters_NrrdCrop ""}

itcl_class Teem_Filters_NrrdCrop {
    inherit Module
    constructor {config} {
        set name NrrdCrop
        set_defaults
    }
    method set_defaults {} {
        global $this-minAxis0
        global $this-maxAxis0
        global $this-minAxis1
        global $this-maxAxis1
        global $this-minAxis2
        global $this-maxAxis2
	global $this-minAxis3
        global $this-maxAxis3
	global $this-absmaxAxis0
	global $this-absmaxAxis1
	global $this-absmaxAxis2
	global $this-absmaxAxis3
        set $this-minAxis0 0
        set $this-maxAxis0 127
	set $this-absmaxAxis0 127
        set $this-minAxis1 0
        set $this-maxAxis1 127
	set $this-absmaxAxis1 127
        set $this-minAxis2 0
        set $this-maxAxis2 127
	set $this-absmaxAxis2 127
	set $this-minAxis3 0
        set $this-maxAxis3 127
	set $this-absmaxAxis3 127
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

    method set_max_vals {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    set_scale_max_value $w.f.ta [set $this-absmaxAxis0]
	    set $this-maxAxis0 [set $this-absmaxAxis0]
	    set_scale_max_value $w.f.a1 [set $this-absmaxAxis1]
	    set $this-maxAxis1 [set $this-absmaxAxis1]
	    set_scale_max_value $w.f.a2 [set $this-absmaxAxis2]
	    set $this-maxAxis2 [set $this-absmaxAxis2]
	    set_scale_max_value $w.f.a3 [set $this-absmaxAxis3]
	    set $this-maxAxis3 [set $this-absmaxAxis3]
	}
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
	min_max_widget $w.f.ta "Tuple Axis" \
	    $this-minAxis0 $this-maxAxis0 $this-absmaxAxis0
	min_max_widget $w.f.a1 Axis1 \
	    $this-minAxis1 $this-maxAxis1 $this-absmaxAxis1
	min_max_widget $w.f.a2 Axis2 \
	    $this-minAxis2 $this-maxAxis2 $this-absmaxAxis2
	min_max_widget $w.f.a3 Axis3 \
	    $this-minAxis3 $this-maxAxis3 $this-absmaxAxis3

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
