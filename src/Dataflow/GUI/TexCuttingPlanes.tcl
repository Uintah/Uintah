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

package require Iwidgets 3.0  

catch {rename TexCuttingPlanes ""}

itcl_class SCIRun_Visualization_TexCuttingPlanes {
    inherit Module

    #protected inactivecolor "#010101"
    constructor {config} {
	set name TexCuttingPlanes
	set_defaults
    }
    method set_defaults {} {
	global $this-control_pos_saved
	global $this-control_x
	global $this-control_y
	global $this-control_z
	global $this-drawX
	global $this-drawY
	global $this-drawZ
	global $this-drawView
	global $this-interp_mode 
	global standard
	global $this-phi_0
	global $this-phi_1
	global $this-draw_phi_0
	global $this-draw_phi_1
	global $this-cyl_active
	set $this-drawX 0
	set $this-drawY 0
	set $this-drawZ 0
	set $this-drawView 0
	set $this-draw_phi_0 0
	set $this-draw_phi_1 0
	set $this-interp_mode 1
	set $this-phi_0 30.0
	set $this-phi_1 60.0
	set $this-control_pos_saved 0
    }


    method set_active_tab {act} {
	global $this-cyl_active
	if {$act == "Cylindrical"} {
	    set $this-cyl_active 1
	} else {
	    set $this-cyl_active 0
	}
	$this-c needexecute
    }

    method spin_in {new phi} {
	if {! [regexp "\\A\\d*\\.*\\d+\\Z" $new]} {
	    return 0
	} elseif {$new > 360.0 || $new < 0.0} {
	    return 0
	} 
	set $phi $new
	$this-c needexecute
	return 1
    }

    method spin_angle {step cyl phi} {
	set newphi [expr [set $phi] + 5 * $step]

	if {$newphi > 360.0} {
	    set newphi [expr $newphi - 360.0]
	} elseif {$newphi < 0.0} {
	    set newphi [expr 360.0 + $newphi]
	}   
	set $phi $newphi
	$cyl delete 0 end
	$cyl insert 0 [set $phi]
	$this-c needexecute
    }
	
    method add_cyl_tab {cyl} {
	set act "$this state"
	set n "$this-c needexecute "
	global $this-phi_0
	global $this-phi_1
	
	label $cyl.l -text "Select Plane(s)"
	checkbutton $cyl.xp -text "Phi-0 plane" -relief flat \
		-variable $this-draw_phi_0 -onvalue 1 -offvalue 0 \
		-anchor w -command "$n"

	checkbutton $cyl.yp -text "Phi-1 plane" -relief flat \
		-variable $this-draw_phi_1 -onvalue 1 -offvalue 0 \
		-anchor w -command "$n"
	
	checkbutton $cyl.zp -text "Z plane" -relief flat \
		-variable $this-drawZ -onvalue 1 -offvalue 0 \
		-anchor w -command "set $this-drawView 0; $act; $n"
	pack $cyl.l $cyl.xp $cyl.yp $cyl.zp \
		-side top -fill x


	iwidgets::spinner $cyl.sp0 -labeltext "Phi-0 degrees: " \
		-width 10 -fixed 10 \
		-validate "$this spin_in %P $this-phi_0" \
		-decrement "$this spin_angle -1 $cyl.sp0 $this-phi_0" \
		-increment "$this spin_angle 1 $cyl.sp0 $this-phi_0" 

	$cyl.sp0 insert 0 [set $this-phi_0]

	iwidgets::spinner $cyl.sp1 -labeltext "Phi-1 degrees: " \
		-width 10 -fixed 10 \
		-validate "$this spin_in %P $this-phi_1" \
		-decrement "$this spin_angle -1 $cyl.sp1 $this-phi_1" \
		-increment "$this spin_angle 1 $cyl.sp1 $this-phi_1" 

	$cyl.sp1 insert 0 [set $this-phi_1]
	
	pack $cyl.sp0 -side top -fill x
	pack $cyl.sp1 -side top -fill x

	frame $cyl.z
	button $cyl.z.plus -text "+" -command "$this-c MoveWidget zplus 1; $n"
	label $cyl.z.label -text " Z "
	button $cyl.z.minus -text "-" -command "$this-c MoveWidget zplus -1; $n"
	pack $cyl.z.plus $cyl.z.label $cyl.z.minus -side left -fill x -expand 1
	pack $cyl.z -side top -fill x -expand 1


    }
    method add_standard_tab {standard} {
	global $this-render_style
	set act "$this state"
	set n "$this-c needexecute "

	label $standard.l -text "Select Plane(s)"
	checkbutton $standard.xp -text "X plane" -relief flat \
		-variable $this-drawX -onvalue 1 -offvalue 0 \
		-anchor w -command "set $this-drawView 0; $act; $n"
	
	checkbutton $standard.yp -text "Y plane" -relief flat \
		-variable $this-drawY -onvalue 1 -offvalue 0 \
		-anchor w -command "set $this-drawView 0; $act; $n"
	
	checkbutton $standard.zp -text "Z plane" -relief flat \
		-variable $this-drawZ -onvalue 1 -offvalue 0 \
		-anchor w -command "set $this-drawView 0; $act; $n"
	
	checkbutton $standard.vp -text "V (view) plane" -relief flat \
		-variable $this-drawView -onvalue 1 -offvalue 0 \
		-anchor w -command \
		"set $this-drawX 0; set $this-drawY 0; set $this-drawZ 0; $act; $n"
	
	
	pack $standard.l $standard.xp $standard.yp $standard.zp $standard.vp \
		-side top -fill x
	
	frame $standard.x
	button $standard.x.plus -text "+" -command "$this-c MoveWidget xplus 1; $n"
	label $standard.x.label -text " X "
	button $standard.x.minus -text "-" -command "$this-c MoveWidget xplus -1; $n"
	pack $standard.x.plus $standard.x.label $standard.x.minus -side left -fill x -expand 1
	pack $standard.x -side top -fill x -expand 1
	
	frame $standard.y
	button $standard.y.plus -text "+" -command "$this-c MoveWidget yplus 1; $n"
	label $standard.y.label -text " Y "
	button $standard.y.minus -text "-" -command "$this-c MoveWidget yplus -1; $n"
	pack $standard.y.plus $standard.y.label $standard.y.minus -side left -fill x -expand 1
	pack $standard.y -side top -fill x -expand 1
	
	frame $standard.z
	button $standard.z.plus -text "+" -command "$this-c MoveWidget zplus 1; $n"
	label $standard.z.label -text " Z "
	button $standard.z.minus -text "-" -command "$this-c MoveWidget zplus -1; $n"
	pack $standard.z.plus $standard.z.label $standard.z.minus -side left -fill x -expand 1
	pack $standard.z -side top -fill x -expand 1
	
	frame $standard.v
	button $standard.v.plus -text "+" -command "$this-c MoveWidget vplus 1; $n"
	label $standard.v.label -text " V "
	button $standard.v.minus -text "-" -command "$this-c MoveWidget vplus -1; $n"
	pack $standard.v.plus $standard.v.label $standard.v.minus -side left -fill x -expand 1
	pack $standard.v -side top -fill x -expand 1
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this-c needexecute "
	#wm minsize $w 250 300
	iwidgets::labeledframe $w.frame_title \
		-labelpos nw -labeltext "Plane Options"
	set dof [$w.frame_title childsite]

	iwidgets::tabnotebook  $dof.tabs -height 250 -raiseselect true 

	global standard
	set st "Standard"
	set standard [$dof.tabs add -label $st \
		-command "$this set_active_tab $st"]

	add_standard_tab $standard

	global cyl
	set c "Cylindrical"
	set cyl [$dof.tabs add -label $c \
		-command "$this set_active_tab $c"]

	add_cyl_tab $cyl
 
	$dof.tabs view "Standard"	
	$dof.tabs configure -tabpos "n"
	pack $dof.tabs -side top -expand yes

	pack $w.frame_title -side top -expand yes

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
	global standard

	if {[set $this-drawX] == 1} {
	    $this activate $standard.x.plus 
	    $standard.x.label configure -foreground black
	    $this  activate $standard.x.minus
	} else {
	   $this  deactivate $standard.x.plus 
	    $standard.x.label configure -foreground $inactivecolor
	    $this deactivate $standard.x.minus
	}

	if {[set $this-drawY] == 1} {
	    $this activate $standard.y.plus 
	    $standard.y.label configure -foreground black
	    $this activate $standard.y.minus
	} else {
	   $this  deactivate $standard.y.plus 
	    $standard.y.label configure -foreground $inactivecolor
	    $this deactivate $standard.y.minus
	}
	if {[set $this-drawZ] == 1} {
	    $this activate $standard.z.plus 
	    $standard.z.label configure -foreground black
	    $this activate $standard.z.minus
	} else {
	    $this deactivate $standard.z.plus 
	    $standard.z.label configure -foreground $inactivecolor
	    $this deactivate $standard.z.minus
	}
	if {[set $this-drawView] == 1} {
	    $this activate $standard.v.plus 
	    $standard.v.label configure -foreground black
	    $this activate $standard.v.minus
	} else {
	    $this deactivate $standard.v.plus 
	    $standard.v.label configure -foreground $inactivecolor
	    $this deactivate $standard.v.minus
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
