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


catch {rename TexCuttingPlanes ""}

itcl_class SCIRun_Visualization_TexCuttingPlanes {
    inherit Module

    #protected inactivecolor "#010101"
    constructor {config} {
	set name TexCuttingPlanes
	set_defaults
    }
    method set_defaults {} {
	global $this-drawX
	global $this-drawY
	global $this-drawZ
	global $this-drawView
	global $this-interp_mode 
	set $this-drawX 0
	set $this-drawY 0
	set $this-drawZ 0
	set $this-drawView 0
	set $this-interp_mode 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 250 300
	frame $w.f -relief groove -borderwidth 2 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "
	set act "$this state"

	global $this-render_style
	label $w.f.l -text "Select Plane(s)"
	checkbutton $w.f.xp -text "X plane" -relief flat \
	    -variable $this-drawX -onvalue 1 -offvalue 0 \
	    -anchor w -command "set $this-drawView 0; $act; $n"

	checkbutton $w.f.yp -text "Y plane" -relief flat \
	    -variable $this-drawY -onvalue 1 -offvalue 0 \
	    -anchor w -command "set $this-drawView 0; $act; $n"

	checkbutton $w.f.zp -text "Z plane" -relief flat \
	    -variable $this-drawZ -onvalue 1 -offvalue 0 \
	    -anchor w -command "set $this-drawView 0; $act; $n"

	checkbutton $w.f.vp -text "V (view) plane" -relief flat \
	    -variable $this-drawView -onvalue 1 -offvalue 0 \
	    -anchor w -command \
	    "set $this-drawX 0; set $this-drawY 0; set $this-drawZ 0; $act; $n"


	pack $w.f.l $w.f.xp $w.f.yp $w.f.zp $w.f.vp \
		-side top -fill x

	frame $w.f.x
	button $w.f.x.plus -text "+" -command "$this-c MoveWidget xplus 1; $n"
	label $w.f.x.label -text " X "
	button $w.f.x.minus -text "-" -command "$this-c MoveWidget xplus -1; $n"
	pack $w.f.x.plus $w.f.x.label $w.f.x.minus -side left -fill x -expand 1
	pack $w.f.x -side top -fill x -expand 1

	frame $w.f.y
	button $w.f.y.plus -text "+" -command "$this-c MoveWidget yplus 1; $n"
	label $w.f.y.label -text " Y "
	button $w.f.y.minus -text "-" -command "$this-c MoveWidget yplus -1; $n"
	pack $w.f.y.plus $w.f.y.label $w.f.y.minus -side left -fill x -expand 1
	pack $w.f.y -side top -fill x -expand 1

	frame $w.f.z
	button $w.f.z.plus -text "+" -command "$this-c MoveWidget zplus 1; $n"
	label $w.f.z.label -text " Z "
	button $w.f.z.minus -text "-" -command "$this-c MoveWidget zplus -1; $n"
	pack $w.f.z.plus $w.f.z.label $w.f.z.minus -side left -fill x -expand 1
	pack $w.f.z -side top -fill x -expand 1

	frame $w.f.v
	button $w.f.v.plus -text "+" -command "$this-c MoveWidget vplus 1; $n"
	label $w.f.v.label -text " V "
	button $w.f.v.minus -text "-" -command "$this-c MoveWidget vplus -1; $n"
	pack $w.f.v.plus $w.f.v.label $w.f.v.minus -side left -fill x -expand 1
	pack $w.f.v -side top -fill x -expand 1

	frame $w.f3 -relief groove -borderwidth 2
	pack $w.f3 -padx 2 -pady 2 -fill x

	label $w.f3.l -text "Interpolation Mode"
	radiobutton $w.f3.interp -text "Interpolate" -relief flat \
		-variable $this-interp_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.f3.near -text "Nearest" -relief flat \
		-variable $this-interp_mode -value 0 \
		-anchor w -command $n

	pack $w.f3.l $w.f3.interp $w.f3.near \
		-side top -fill x
	
	button $w.exec -text "Execute" -command $n
	pack $w.exec -side top -fill x
	
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -fill x

	$this state 
    }


    method state {} {
	set inactivecolor "#010101"
	set w .ui[modname]
	if {[set $this-drawX] == 1} {
	    $this activate $w.f.x.plus 
	    $w.f.x.label configure -foreground black
	    $this  activate $w.f.x.minus
	} else {
	   $this  deactivate $w.f.x.plus 
	    $w.f.x.label configure -foreground $inactivecolor
	    $this deactivate $w.f.x.minus
	}

	if {[set $this-drawY] == 1} {
	    $this activate $w.f.y.plus 
	    $w.f.y.label configure -foreground black
	    $this activate $w.f.y.minus
	} else {
	   $this  deactivate $w.f.y.plus 
	    $w.f.y.label configure -foreground $inactivecolor
	    $this deactivate $w.f.y.minus
	}
	if {[set $this-drawZ] == 1} {
	    $this activate $w.f.z.plus 
	    $w.f.z.label configure -foreground black
	    $this activate $w.f.z.minus
	} else {
	    $this deactivate $w.f.z.plus 
	    $w.f.z.label configure -foreground $inactivecolor
	    $this deactivate $w.f.z.minus
	}
	if {[set $this-drawView] == 1} {
	    $this activate $w.f.v.plus 
	    $w.f.v.label configure -foreground black
	    $this activate $w.f.v.minus
	} else {
	    $this deactivate $w.f.v.plus 
	    $w.f.v.label configure -foreground $inactivecolor
	    $this deactivate $w.f.v.minus
	}

    }
    method activate { w } {
	$w configure -state normal -foreground black
    }
    method deactivate { w } {
	set inactivecolor "#010101"
	$w configure -state disabled -foreground $inactivecolor
    }
}
