#
#  FaceCuttingPlane.tcl
#
#  Written by:
#   Kurt Zimmerman
#   SCI Institute
#   University of Utah
#   August 2002
#
#  Copyright (C) 2002 SCI Group
#

itcl_class Uintah_Visualization_FaceCuttingPlane {
    inherit Module
    constructor {config} {
	set name FaceCuttingPlane
	set_defaults
    }
    method set_defaults {} {
	global $this-face_name
	global $this-where
	global $this-line_size
	global $this-need_find
	set $this-face_name ""
	set $this-where 0.5
	set $this-line_size 1
	set $this-need_find 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 20
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x -side top
	set n "$this-c needexecute "

	# for disabled foreground
	set color "#505050"  

	label $w.f.fn -textvariable "$this-face_name"
	pack $w.f.fn -padx 2 -pady 2 -side top -anchor center
	button $w.f.findxy -text "Find XY" -command "$this-c findxy" \
	    -disabledforeground $color
	pack $w.f.findxy -pady 2 -side top -ipadx 3 -anchor center
	
	button $w.f.findyz -text "Find YZ" -command "$this-c findyz" \
	    -disabledforeground $color
	pack $w.f.findyz -pady 2 -side top -ipadx 3 -anchor center
	
	button $w.f.findxz -text "Find XZ" -command "$this-c findxz" \
	    -disabledforeground $color
	pack $w.f.findxz -pady 2 -side top -ipadx 3 -anchor center

	frame $w.g -bd 2 -relief groove
	frame $w.g.x
	button $w.g.x.plus -text "+" -command "$this-c MoveWidget plusx" \
	    -disabledforeground $color
	label $w.g.x.label -text " X " -disabledforeground $color
	button $w.g.x.minus -text "-" -command "$this-c MoveWidget minusx" \
	    -disabledforeground $color
	pack $w.g.x.plus $w.g.x.label $w.g.x.minus -side left -fill x -expand 1 -padx 10
	pack $w.g.x -side top

	frame $w.g.y
	button $w.g.y.plus -text "+" -command "$this-c MoveWidget plusy" \
	    -disabledforeground $color
	label $w.g.y.label -text " Y " -disabledforeground $color
	button $w.g.y.minus -text "-" -command "$this-c MoveWidget minusy" \
	    -disabledforeground $color
	pack $w.g.y.plus $w.g.y.label $w.g.y.minus -side left -fill x -expand 1 -padx 10
	pack $w.g.y -side top

	frame $w.g.z
	button $w.g.z.plus -text "+" -command "$this-c MoveWidget plusz" \
	    -disabledforeground $color
	label $w.g.z.label -text " Z " -disabledforeground $color
	button $w.g.z.minus -text "-" -command "$this-c MoveWidget minusz" \
	    -disabledforeground $color
	pack $w.g.z.plus $w.g.z.label $w.g.z.minus -side left \
	    -fill x -expand 1 -padx 10
	pack $w.g.z -side top
	pack $w.g -side top -fill x

	frame $w.ls -bd 2 -relief groove
	pack $w.ls -side top -fill x -pady 2
	label $w.ls.l -text "Line Size (in pixels)" 
	pack $w.ls.l -side top -anchor center -padx 2 -pady 2
	scale $w.ls.s -variable "$this-line_size" -orient horiz \
	    -from 1 -to 4
	pack $w.ls.s -side top -fill x -padx 2 -pady 2

        button $w.b -text "Close" -command "wm withdraw $w"
        pack $w.b -side top -fill x -padx 2 -pady 2

	bind $w.ls.s <ButtonRelease> $n
	$this update_control
	
    }
    
    method update_control {} {
	set w .ui[modname]

	if {![winfo exists $w]} {
	    return
	}

	if {[set $this-face_name] == "" } { 
	    return 
	}
	

	if { [set $this-face_name] == "X faces" } {
	    if { [set $this-need_find] == 1 } {
		$this enable_nav_button $w.g.z
		$this disable_nav_button $w.g.y
		$this disable_nav_button $w.g.x
	    } else {
		$this enable_nav_button $w.g.y
		$this disable_nav_button $w.g.z
		$this disable_nav_button $w.g.x
	    }
	     $w.f.findxy configure -state normal
	     $w.f.findyz configure -state disabled
	     $w.f.findxz configure -state normal
	} elseif {[set $this-face_name] == "Y faces" } {
	    if { [set $this-need_find] == 1 } {
		$this enable_nav_button $w.g.z
		$this disable_nav_button $w.g.x
		$this disable_nav_button $w.g.y
	    } else {
		$this enable_nav_button $w.g.x
		$this disable_nav_button $w.g.z
		$this disable_nav_button $w.g.y
	    }
	     $w.f.findxy configure -state normal
	     $w.f.findyz configure -state normal
	     $w.f.findxz configure -state disabled
	} else {
	    if { [set $this-need_find] == 2 } {
		$this enable_nav_button $w.g.x
		$this disable_nav_button $w.g.z
		$this disable_nav_button $w.g.y
	    } else {
		$this enable_nav_button $w.g.y
		$this disable_nav_button $w.g.z
		$this disable_nav_button $w.g.x
	    }
	     $w.f.findxy configure -state disabled
	     $w.f.findyz configure -state normal
	     $w.f.findxz configure -state normal
	}   
    }

    method disable_nav_button { b } {
	$b.plus configure -state disabled
	$b.label configure -state disabled
	$b.minus configure -state disabled
    }
    method enable_nav_button { b } {
	$b.plus configure -state normal
	$b.label configure -state normal
	$b.minus configure -state normal
    }
}
