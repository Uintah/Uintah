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

itcl_class SCIRun_Visualization_RescaleColorMap { 
    inherit Module 

    constructor {config} { 
        set name RescaleColorMap 
        set_defaults 
    } 
  
    method set_defaults {} { 
	global $this-isFixed
	global $this-min
	global $this-max
	global $this-makeSymmetric
	set bVar 0
	set $this-isFixed 0
	set $this-min 0
	set $this-max 1
	set $this-makeSymmetric 0
    }   

    method ui {} { 
	global $this-isFixed
	global $this-min
	global $this-max
	global $this-makeSymmetric

	set w .ui[modname]
	
	if {[winfo exists $w]} { 
	    wm deiconify $w
	    raise $w 
	    return; 
	} 
	
	toplevel $w 
	wm minsize $w 200 50 
 
	frame $w.f1
	
	frame $w.f1.a -relief flat
	pack $w.f1.a -side left -expand yes -fill x
	radiobutton $w.f1.a.b -text "Auto Scale" -variable $this-isFixed \
	    -value 0 -command "$this autoScale"
	pack $w.f1.a.b -side left

	frame $w.f1.s -relief flat
	pack $w.f1.s -side top -expand yes -fill x
	checkbutton $w.f1.s.b -text "Symmetric Auto Scale" \
	    -variable $this-makeSymmetric
	pack $w.f1.s.b -side left
	
	pack $w.f1 -side top -expand yes -fill x

	frame $w.f2 -relief flat
	pack $w.f2 -side top -expand yes -fill x
	radiobutton $w.f2.b -text "Fixed Scale"  -variable $this-isFixed \
	    -value 1 -command "$this fixedScale"
	pack $w.f2.b -side left

	frame $w.f3 -relief flat
	pack $w.f3 -side top -expand yes -fill x
	
	label $w.f3.l1 -text "min:  "
	entry $w.f3.e1 -textvariable $this-min

	label $w.f3.l2 -text "max:  "
	entry $w.f3.e2 -textvariable $this-max
	pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
	    -expand yes -fill x -padx 2 -pady 2

	bind $w.f3.e1 <Return> "$this-c needexecute"
	bind $w.f3.e2 <Return> "$this-c needexecute"

	button $w.execute -text Execute -command "$this-c needexecute"
	button $w.close -text Close -command "destroy $w"
	pack $w.execute $w.close -side left  -padx 5 -expand 1 -fill x

	if { [set $this-isFixed] } {
	    $w.f2.b select
	    $this fixedScale
	} else {
	    $w.f1.a.b select
	    $this autoScale
	}
    }

    method autoScale { } {
	global $this-isFixed
	set w .ui[modname]
	
	set color "#505050"

	$w.f1.s.b configure -state normal
	$w.f3.l1 configure -foreground $color
	$w.f3.e1 configure -state disabled -foreground $color
	$w.f3.l2 configure -foreground $color
	$w.f3.e2 configure -state disabled -foreground $color
    }

    method fixedScale { } {
	global $this-isFixed
	set w .ui[modname]

	$w.f1.s.b configure -state disabled
	$w.f3.l1 configure -foreground black
	$w.f3.e1 configure -state normal -foreground black
	$w.f3.l2 configure -foreground black
	$w.f3.e2 configure -state normal -foreground black
    }
}
