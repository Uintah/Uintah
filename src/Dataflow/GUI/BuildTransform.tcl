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

catch {rename BuildTransform ""}

itcl_class SCIRun_Math_BuildTransform {
    inherit Module
    constructor {config} {
	set name BuildTransform
	set_defaults
    }
    method set_defaults {} {
	global $this-rotate_x
	global $this-rotate_y
	global $this-rotate_z
	global $this-rotate_theta
	global $this-scale_uniform
	global $this-scale_x
	global $this-scale_y
	global $this-scale_z
	global $this-translate_x
	global $this-translate_y
	global $this-translate_z
	global $this-shear_plane_a
	global $this-shear_plane_b
	global $this-shear_plane_c
	global $this-shear_plane_d
	global $this-widget_resizable
	global $this-pre_transform
	global $this-last_transform
	global $this-which_transform
	global $this-permute_x
	global $this-permute_y
	global $this-permute_z
	global $this-ignoring_widget_changes
	global $this-widgetScale
	global $this-loginput
	global $this-logoutput
	set $this-rotate_x 0
	set $this-rotate_y 0
	set $this-rotate_z 1
	set $this-rotate_theta 0
	set $this-scale_uniform 0
	set $this-scale_x 0
	set $this-scale_y 0
	set $this-scale_z 0
	set $this-translate_x 0
	set $this-translate_y 0
	set $this-translate_z 0
	set $this-shear_plane_a 0
	set $this-shear_plane_b 0
	set $this-shear_plane_c 1
	set $this-shear_plane_a 0
	set $this-last_transform translate
	set $this-which_transform translate
	set $this-pre_transform 1
	set $this-permute_x 1
	set $this-permute_y 2
	set $this-permute_z 3
	set $this-widget_scale 1
	set $this-widget_resizable 1
	set $this-ignoring_widget_changes 1
	set $this-loginput 100.0
	set $this-logoutput 2.0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}

	toplevel $w
	wm minsize $w 200 50

	set mouseXLoc [winfo pointerx .]
	set mouseYLoc [winfo pointery .]
	wm geometry $w +$mouseXLoc+$mouseYLoc

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2

	global $this-translate_x
	global $this-translate_y
	global $this-translate_z
	global $this-pre_transform
	global $this-which_transform
    
	frame $w.f.which
	radiobutton $w.f.which.trans \
		-command "$this set_transform $w translate" \
		-text Translate -variable $this-which_transform \
		-value "translate"
	radiobutton $w.f.which.scale \
		-command "$this set_transform $w scale" \
		-text Scale -variable $this-which_transform \
		-value "scale"
	radiobutton $w.f.which.rot \
		-command "$this set_transform $w rotate" \
		-text Rotate -variable $this-which_transform \
		-value "rotate"
	radiobutton $w.f.which.shear \
		-command "$this set_transform $w shear" \
		-text Shear -variable $this-which_transform \
		-value "shear"
	radiobutton $w.f.which.permute \
		-command "$this set_transform $w permute" \
		-text Permute -variable $this-which_transform \
		-value "permute"
	radiobutton $w.f.which.widget \
		-command "$this set_transform $w widget" \
		-text Widget -variable $this-which_transform \
		-value "widget"
	pack $w.f.which.trans $w.f.which.scale $w.f.which.rot \
		$w.f.which.shear $w.f.which.permute $w.f.which.widget \
		-side left -fill x -expand 1

	frame $w.f.b 
	button $w.f.b.doit -text "Apply Transform" \
		-command "$this-c needexecute"
	button $w.f.b.comp -text "Composite Transform" \
		-command "$this-c composite; $this set_transform $w translate; $this set_defaults"
	button $w.f.b.reset -text "Reset" \
		-command "$this-c reset; $this set_transform $w translate; $this set_defaults"
	pack $w.f.b.doit $w.f.b.comp $w.f.b.reset \
		-side left -fill x -padx 10 -pady 3

	frame $w.f.prepost
	radiobutton $w.f.prepost.pre -variable $this-pre_transform \
		-text "Pre-multiply" -value 0
	radiobutton $w.f.prepost.post -variable $this-pre_transform \
		-text "Post-multiply" -value 1
	pack $w.f.prepost.pre $w.f.prepost.post -side left \
		-fill x -expand 1

	pack $w.f.which -side top -fill x -expand 1
	pack $w.f.b -side top
	pack $w.f.prepost -side top -fill x -expand 1

	frame $w.f.t -relief groove -borderwidth 2
	label $w.f.t.l -text "Translate Vector"
	frame $w.f.t.f
	expscale $w.f.t.f.x -orient horizontal -variable $this-translate_x \
		-label "X:"
	expscale $w.f.t.f.y -orient horizontal -variable $this-translate_y \
		-label "Y:"
	expscale $w.f.t.f.z -orient horizontal -variable $this-translate_z \
		-label "Z:"
	pack $w.f.t.f.x $w.f.t.f.y $w.f.t.f.z -side top -fill x
	pack $w.f.t.l $w.f.t.f -side top -fill both -expand 1 -padx 2 -pady 2
	pack $w.f.t -side top -fill x -expand 1
	
	global $this-rotate_x
	global $this-rotate_y
	global $this-rotate_z
	global $this-rotate_theta
	frame $w.f.r -relief groove -borderwidth 5
	scale $w.f.r.x -orient horizontal -variable $this-rotate_x \
		-label "Rotate Axis X:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.r.y -orient horizontal -variable $this-rotate_y \
		-label "Rotate Axis Y:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.r.z -orient horizontal -variable $this-rotate_z \
		-label "Rotate Axis Z:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.r.th -orient horizontal -variable $this-rotate_theta \
		-label "Rotate Theta (degrees):" -showvalue true -from -360.0 \
		-to 360.0 -resolution 0.1
	pack $w.f.r.x $w.f.r.y $w.f.r.z $w.f.r.th -fill x -expand 1 -side top

	global $this-xstr
	global $this-ystr
	global $this-zstr
	$this build_map_string
	frame $w.f.p -relief groove -borderwidth 5
	label $w.f.p.l -text "Field Map"
	pack $w.f.p.l -side top -fill both -expand 1
	frame $w.f.p.m -relief sunken -bd 2
	frame $w.f.p.m.l
	button $w.f.p.m.l.x -text "FlipX" -command "$this flip_x" -padx 8
	button $w.f.p.m.l.y -text "FlipY" -command "$this flip_y" -padx 8
	button $w.f.p.m.l.z -text "FlipZ" -command "$this flip_z" -padx 8
	pack $w.f.p.m.l.x $w.f.p.m.l.y $w.f.p.m.l.z -side top
	pack $w.f.p.m.l -side left -expand 1 -fill x
	frame $w.f.p.m.r
	label $w.f.p.m.r.x -textvariable $this-xstr
	label $w.f.p.m.r.y -textvariable $this-ystr
	label $w.f.p.m.r.z -textvariable $this-zstr
	pack $w.f.p.m.r.x $w.f.p.m.r.y $w.f.p.m.r.z -side top
	pack $w.f.p.m.r -side left -expand 1 -fill x
	pack $w.f.p.m -side top -fill x -expand 1
	frame $w.f.p.b -relief sunken -bd 2
	frame $w.f.p.b.l
	frame $w.f.p.b.r
	button $w.f.p.b.l.cp -text "Cycle+" -command "$this cycle_pos"
	button $w.f.p.b.l.cn -text "Cycle-" -command "$this cycle_neg"
	button $w.f.p.b.l.res -text "Reset" -command "$this reset"
	button $w.f.p.b.r.sxy -text "SwapXY" -command "$this swap_XY"
	button $w.f.p.b.r.syz -text "SwapYZ" -command "$this swap_YZ"
	button $w.f.p.b.r.sxz -text "SwapXZ" -command "$this swap_XZ"
	pack $w.f.p.b.l.cp $w.f.p.b.l.cn $w.f.p.b.l.res \
		-side top -expand 1 -fill both
	pack $w.f.p.b.r.sxy $w.f.p.b.r.syz $w.f.p.b.r.sxz \
		-side top -expand 1 -fill both
	pack $w.f.p.b.l $w.f.p.b.r -side left -expand 1 -fill both
	pack $w.f.p.b -side bottom -fill both -expand 1

	global $this-scale_uniform
	global $this-scale_x
	global $this-scale_y
	global $this-scale_z
	frame $w.f.s
	frame $w.f.s.g -relief groove -borderwidth 5
	label $w.f.s.g.l -text "Log Scale: "
	scale $w.f.s.g.s -variable $this-scale_uniform -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.s.sx -relief groove -borderwidth 5
	label $w.f.s.sx.l -text "Log ScaleX: "
	scale $w.f.s.sx.s -variable $this-scale_x -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.s.sy -relief groove -borderwidth 5
	label $w.f.s.sy.l -text "Log ScaleY: "
	scale $w.f.s.sy.s -variable $this-scale_y -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	frame $w.f.s.sz -relief groove -borderwidth 5
	label $w.f.s.sz.l -text "Log ScaleZ: "
	scale $w.f.s.sz.s -variable $this-scale_z -orient horizontal \
		-from -3.000 -to 3.000 -resolution .001 -showvalue true
	pack $w.f.s.g.l -side left
	pack $w.f.s.g.s -side left -expand 1 -fill x
	pack $w.f.s.sx.l -side left
	pack $w.f.s.sx.s -side left -expand 1 -fill x
	pack $w.f.s.sy.l -side left
	pack $w.f.s.sy.s -side left -expand 1 -fill x
	pack $w.f.s.sz.l -side left
	pack $w.f.s.sz.s -side left -expand 1 -fill x
	frame $w.f.s.e
	label $w.f.s.e.l1 -text "Log Calculator: log("
	global $this-loginput
	entry $w.f.s.e.e1 -textvariable $this-loginput -width 8
	bind $w.f.s.e.e1 <Return> "$this computelog"
	label $w.f.s.e.l2 -text ") = "
	global $this-logoutput
	label $w.f.s.e.l3 -textvariable $this-logoutput -width 12
	pack $w.f.s.e.l1 $w.f.s.e.e1 $w.f.s.e.l2 $w.f.s.e.l3 -side left
	pack $w.f.s.g $w.f.s.sx $w.f.s.sy $w.f.s.sz -side top -fill x -expand 1
	pack $w.f.s.e -side top
	global $this-shear_plane_a
	global $this-shear_plane_b
	global $this-shear_plane_c
	global $this-shear_plane_d
	frame $w.f.sh -relief groove -borderwidth 5
	label $w.f.sh.l -text "Shear Fixed Plane"
	scale $w.f.sh.a -orient horizontal -variable $this-shear_plane_a \
		-label "A:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.sh.b -orient horizontal -variable $this-shear_plane_b \
		-label "B:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	scale $w.f.sh.c -orient horizontal -variable $this-shear_plane_c \
		-label "C:" -showvalue true -from -1.0 -to 1.0 \
		-resolution .01
	expscale $w.f.sh.d -orient horizontal -variable $this-shear_plane_d \
		-label "D:"
	pack $w.f.sh.l $w.f.sh.a $w.f.sh.b $w.f.sh.c -fill x \
		-expand 1 -side top
	pack $w.f.sh.d -fill x \
		-expand 1 -side bottom

	global $this-widget_scale
	global $this-ignoring_widget_changes
	global $this-widget_show_resize_handles
	frame $w.f.w -relief groove -borderwidth 5
	label $w.f.w.l -text "Widget"
	pack $w.f.w.l -side top
	expscale $w.f.w.d -orient horizontal -variable $this-widget_scale \
		-label "Uniform Scale"
	frame $w.f.w.b 

	set $this-widget_scale 1
	set $this-widgetShowResizeHandles 0

	checkbutton $w.f.w.b.handles -text "Resize Separably" \
		-variable $this-widget_resizable \
		-command "$this change_handles"
	button $w.f.w.b.reset -text "Reset Widget" \
		-command "$this-c reset_widget"
	checkbutton $w.f.w.b.ignore -text "Ignore Changes" \
		-variable $this-ignoring_widget_changes
	pack $w.f.w.b.handles $w.f.w.b.reset $w.f.w.b.ignore -side left \
		-fill x -expand 1 -pady 3 -padx 12
	pack $w.f.w.d $w.f.w.b -side top -fill x -expand 1

	pack $w.f -fill x -expand 1 -side top
	$this set_transform $w [set $this-which_transform]

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }	

    method computelog { } {
	global $this-loginput
	global $this-logoutput
	set x [set $this-loginput]
	set $this-logoutput [ expr log10($x) ]
    }

    method change_handles { } {
	global $this-widget_resizable
	$this-c change_handles [set $this-widget_resizable]
    }

    method set_transform {w t} {
	global $this-last_transform
	if {[set $this-last_transform] == $t} return
	if {[set $this-last_transform] == "rotate"} {pack forget $w.f.r}
	if {[set $this-last_transform] == "scale"} {pack forget $w.f.s}
	if {[set $this-last_transform] == "shear"} {pack forget $w.f.sh}
	if {[set $this-last_transform] == "permute"} {
	    pack forget $w.f.p
	    pack $w.f.t -side top -fill x -expand 1
	}
	if {[set $this-last_transform] == "widget"} {
	    pack forget $w.f.w
	    pack $w.f.t -side top -fill x -expand 1
	    $this-c hide_widget
	}
	set $this-last_transform $t
	
	if {$t == "translate"} {
	    $w.f.t.l configure -text "Translate Vector"
	    return
	}
	if {$t == "rotate"} {
	    $w.f.t.l configure -text "Rotation Fixed Point"
	    pack $w.f.r -side top -fill x -expand 1
	    return
	}
	if {$t == "scale"} {
	    $w.f.t.l configure -text "Scale Fixed Point"
	    pack $w.f.s -side top -fill x -expand 1
	    return
	}
	if {$t == "shear"} {
	    $w.f.t.l configure -text "Shear Vector"
	    pack $w.f.sh -side top -fill x -expand 1
	    return
	}
	if {$t == "permute"} {
	    pack forget $w.f.t
	    pack $w.f.p -side top -fill x -expand 1
	    $this-c show_widget
	    return
	}
	if {$t == "widget"} {
	    pack forget $w.f.t
	    pack $w.f.w -side top -fill x -expand 1
	    $this-c show_widget
	}
    }

    method value_to_string { v } {
	if {$v == 1} {
	    return x+
	}
	if {$v == -1} {
	    return x-
	}
	if {$v == 2} {
	    return y+
	}
	if {$v == -2} {
	    return y-
	}
	if {$v == 3} {
	    return z+
	}
	return "z-"
    }

    method build_map_string { } {
	global $this-permute_x
	global $this-permute_y
	global $this-permute_z
	global $this-xstr
	global $this-ystr
	global $this-zstr
	
	set xx [$this value_to_string [set $this-permute_x]]
	set yy [$this value_to_string [set $this-permute_y]]
	set zz [$this value_to_string [set $this-permute_z]]
	set $this-xstr "x <- $xx"
	set $this-ystr "y <- $yy"
	set $this-zstr "z <- $zz"
    }

    method flip_x { } {
	global $this-permute_x
	set $this-permute_x [expr [set $this-permute_x] * -1]
	$this build_map_string
    }
    
    method flip_y { } {
	global $this-permute_y
	set $this-permute_y [expr [set $this-permute_y] * -1]
	$this build_map_string
    }
    
    method flip_z { } {
	global $this-permute_z
	set $this-permute_z [expr [set $this-permute_z] * -1]
	$this build_map_string
    }

    method cycle_pos { } {
	global $this-permute_x
	global $this-permute_y
	global $this-permute_z
	set tmp [set $this-permute_x]
	set $this-permute_x [set $this-permute_y]
	set $this-permute_y [set $this-permute_z]
	set $this-permute_z $tmp
	$this build_map_string
    }

    method cycle_neg { } {
	global $this-permute_x
	global $this-permute_y
	global $this-permute_z
	set tmp [set $this-permute_z]
	set $this-permute_z [set $this-permute_y]
	set $this-permute_y [set $this-permute_x]
	set $this-permute_x $tmp
	$this build_map_string
    }

    method reset { } {
	global $this-permute_x
	global $this-permute_y
	global $this-permute_z
	set $this-permute_x 1
	set $this-permute_y 2
	set $this-permute_z 3
	$this build_map_string
    }

    method swap_XY { } {
	global $this-permute_x
	global $this-permute_y
	set tmp [set $this-permute_x]
	set $this-permute_x [set $this-permute_y]
	set $this-permute_y $tmp
	$this build_map_string
    }

    method swap_XZ { } {
	global $this-permute_x
	global $this-permute_z
	set tmp [set $this-permute_x]
	set $this-permute_x [set $this-permute_z]
	set $this-permute_z $tmp
	$this build_map_string
    }

    method swap_YZ { } {
	global $this-permute_y
	global $this-permute_z
	set tmp [set $this-permute_y]
	set $this-permute_y [set $this-permute_z]
	set $this-permute_z $tmp
	$this build_map_string
    }
}
