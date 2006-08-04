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


catch {rename GridSliceVis ""}

itcl_class Kurt_Visualization_GridSliceVis {
    inherit Module
    constructor {config} {
	set name GridSliceVis
	set_defaults
    }
    method set_defaults {} {

	global $this-max_brick_dim_
	global $this-min_
	global $this-max_
	global $this-is_fixed_
	global $this-drawX
	global $this-drawY
	global $this-drawZ
	global $this-drawView
	global $this-interp_mode 
	global $this-point_x
	global $this-point_y
	global $this-point_z
	global $this-point_init
	set $this-max_brick_dim_ 0
	set $this-min_ 0
	set $this-max_ 1
	set $this-is_fixed_ 0
	set $this-drawX 1
	set $this-drawY 0
	set $this-drawZ 0
	set $this-drawView 0
	set $this-interp_mode 1
	set $this-point_x 0
	set $this-point_y 0
	set $this-point_z 0
	set $this-point_init 0
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
	set n "$this-c needexecute "
	set act "$this state"

	frame $w.dimframe -relief groove -border 2
	label $w.dimframe.l -text "Brick Size Cubed"
	pack $w.dimframe -side top -padx 2 -pady 2 -fill both
	pack $w.dimframe.l -side top -fill x

	if { [set $this-max_brick_dim_] != 0 } {
	    $this SetDims [set $this-max_brick_dim_]
	}
	global $this-is_fixed_
        frame $w.f1 -relief flat
        pack $w.f1 -side top -expand yes -fill x
        radiobutton $w.f1.b -text "Auto Scale"  -variable $this-is_fixed_ \
		-value 0 -command "$this autoScale"
        pack $w.f1.b -side left

        frame $w.f2 -relief flat
        pack $w.f2 -side top -expand yes -fill x
        radiobutton $w.f2.b -text "Fixed Scale"  -variable $this-is_fixed_ \
		-value 1 -command "$this fixedScale"
        pack $w.f2.b -side left

        frame $w.f3 -relief flat
        pack $w.f3 -side top -expand yes -fill x
        
        label $w.f3.l1 -text "min:  "
        entry $w.f3.e1 -textvariable $this-min_

        label $w.f3.l2 -text "max:  "
        entry $w.f3.e2 -textvariable $this-max_
        pack $w.f3.l1 $w.f3.e1 $w.f3.l2 $w.f3.e2 -side left \
            -expand yes -fill x -padx 2 -pady 2

        bind $w.f3.e1 <Return> $n
        bind $w.f3.e2 <Return> $n

	frame $w.f -relief groove -borderwidth 2 
	pack $w.f -padx 2 -pady 2 -fill x

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
	button $w.f.x.plus -text "+" -command "$this-c MoveWidget xplus; $n"
	label $w.f.x.label -text " X "
	button $w.f.x.minus -text "-" -command "$this-c MoveWidget xminus; $n"
	pack $w.f.x.plus $w.f.x.label $w.f.x.minus -side left -fill x -expand 1
	pack $w.f.x -side top -fill x -expand 1

	frame $w.f.y
	button $w.f.y.plus -text "+" -command "$this-c MoveWidget yplus; $n"
	label $w.f.y.label -text " Y "
	button $w.f.y.minus -text "-" -command "$this-c MoveWidget yminus; $n"
	pack $w.f.y.plus $w.f.y.label $w.f.y.minus -side left -fill x -expand 1
	pack $w.f.y -side top -fill x -expand 1

	frame $w.f.z
	button $w.f.z.plus -text "+" -command "$this-c MoveWidget zplus; $n"
	label $w.f.z.label -text " Z "
	button $w.f.z.minus -text "-" -command "$this-c MoveWidget zminus; $n"
	pack $w.f.z.plus $w.f.z.label $w.f.z.minus -side left -fill x -expand 1
	pack $w.f.z -side top -fill x -expand 1

	frame $w.f.v
	button $w.f.v.plus -text "+" -command "$this-c MoveWidget vplus; $n"
	label $w.f.v.label -text " V "
	button $w.f.v.minus -text "-" -command "$this-c MoveWidget vminus; $n"
	pack $w.f.v.plus $w.f.v.label $w.f.v.minus -side left -fill x -expand 1
	pack $w.f.v -side top -fill x -expand 1

	frame $w.f4 -relief groove -borderwidth 2
	pack $w.f4 -padx 2 -pady 2 -fill x

	label $w.f4.l -text "Interpolation Mode"
	radiobutton $w.f4.interp -text "Interpolate" -relief flat \
		-variable $this-interp_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.f4.near -text "Nearest" -relief flat \
		-variable $this-interp_mode -value 0 \
		-anchor w -command $n

	pack $w.f4.l $w.f4.interp $w.f4.near \
		-side top -fill x
	
	button $w.exec -text "Execute" -command $n
	pack $w.exec -side top -fill x
	
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -fill x

	if { [set $this-is_fixed_] } {
            $w.f2.b select
            $this fixedScale
        } else {
            $w.f1.b select
	    global $this-is_fixed_
	    set w .ui[modname]
	    
	    set $this-is_fixed_ 0
	    
	    set color "#505050"
	    
	    $w.f3.l1 configure -foreground $color
	    $w.f3.e1 configure -state disabled -foreground $color
	    $w.f3.l2 configure -foreground $color
	    $w.f3.e2 configure -state disabled -foreground $color
        }

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
    method autoScale { } {
        global $this-is_fixed_
        set w .ui[modname]
        
        set $this-is_fixed_ 0

        set color "#505050"

        $w.f3.l1 configure -foreground $color
        $w.f3.e1 configure -state disabled -foreground $color
        $w.f3.l2 configure -foreground $color
        $w.f3.e2 configure -state disabled -foreground $color
	$this-c needexecute	

    }

    method fixedScale { } {
        global $this-is_fixed_
        set w .ui[modname]

        set $this-is_fixed_ 1


        $w.f3.l1 configure -foreground black
        $w.f3.e1 configure -state normal -foreground black
        $w.f3.l2 configure -foreground black
        $w.f3.e2 configure -state normal -foreground black
        
    }

    method SetDims { val } {
	global $this-max_brick_dim_
	set $this-max_brick_dim_ $val
	set w .ui[modname]

	set vals  [format "%i %i %i %i" [expr $val/8] [expr $val/4] [expr $val/2] $val] 
	set vals [split $vals]
	if {![winfo exists $w]} {
	    return
	}
	if {[winfo exists $w.dimframe.f]} {
	    destroy $w.dimframe.f
	}

	frame $w.dimframe.f -relief flat
	pack $w.dimframe.f -side top -fill x
	set f $w.dimframe.f
	for {set i 0} {$i < 4} { incr i} {
	    set v [lindex $vals $i]
	    radiobutton $f.brickdim$v -text $v -relief flat \
		-variable $this-max_brick_dim_ -value $v \
		-command "$this-c needexecute"
	    pack $f.brickdim$v -side left -padx 2 -fill x
	}
    }
}
