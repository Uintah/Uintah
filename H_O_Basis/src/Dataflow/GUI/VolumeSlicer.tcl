#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


catch {rename VolumeSlicer ""}

itcl_class SCIRun_Visualization_VolumeSlicer {
    inherit Module
    constructor {config} {
	set name VolumeSlicer
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
	global $this-use_stencil
	global $this-multi_level
	global $this-outline_levels
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
	set $this-use_stencil 0
	set $this-multi_level 1
	set $this-outline_levels 0
    }


    method set_active_tab {act} {
	global $this-cyl_active
	if {$act == "Cylindrical"} {
	    set $this-cyl_active 1
	} else {
	    set $this-cyl_active 0
	}
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
            return
        } else {
	    buildTopLevel
	}
    }

    method buildTopLevel {} {
        set w .ui[modname] 

        if {[winfo exists $w]} { 
            return
        } 
	
        toplevel $w 
	wm withdraw $w

	build_ui
    }
     
    method build_ui {} {
	set w .ui[modname]
	set n "$this-c needexecute "
	frame $w.main -relief flat
	pack $w.main -fill both -expand yes

	#wm minsize $w 250 300
	iwidgets::labeledframe $w.main.frame_title \
		-labelpos nw -labeltext "Plane Options"
	set dof [$w.main.frame_title childsite]

	iwidgets::tabnotebook  $dof.tabs -height 250 -raiseselect true 

	global standard
	set st "Standard"
	set standard [$dof.tabs add -label $st \
			  -command "$this set_active_tab $st"]

	add_standard_tab $standard

	global cyl
	set c "Cylindrical"
	set cyl [$dof.tabs add -label $c \
		     -command "$this set_active_tab $c; $n"]

	add_cyl_tab $cyl
	$dof.tabs view "Standard"	
	$dof.tabs configure -tabpos "n"
	$dof.tabs pageconfigure 0 -command "$this set_active_tab $st; $n"
	pack $dof.tabs -side top -expand yes

	pack $w.main.frame_title -side top -expand yes

#  	checkbutton $w.main.cb -text "use stencil" \
#  	    -variable $this-use_stencil -command $n
#  	pack $w.main.cb -side top
	frame $w.main.f4 -relief flat -borderwidth 0
	pack $w.main.f4 -fill x -expand yes

	if { [set $this-multi_level] > 1 } {
	    $this build_multi_level
	}
	
	frame $w.main.f3 -relief groove -borderwidth 2
	pack $w.main.f3 -padx 4 -pady 4 -fill x -side top

	label $w.main.f3.l -text "Interpolation Mode"
	radiobutton $w.main.f3.interp -text "Interpolate" -relief flat \
		-variable $this-interp_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.main.f3.near -text "Nearest" -relief flat \
		-variable $this-interp_mode -value 0 \
		-anchor w -command $n

	pack $w.main.f3.l $w.main.f3.interp $w.main.f3.near -side top -fill x -padx 4 -pady 2
	
	makeSciButtonPanel $w.main $w $this
	$this state 
	moveToCursor $w


	
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

    method build_multi_level { } {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    puts -nonewline "building ml frame"
	    frame $w.main.f4.f -relief groove -borderwidth 2
	    pack $w.main.f4.f -padx 2 -pady 2 -fill x -expand yes -side top
	    frame $w.main.f4.f.f1 -relief flat -borderwidth 2
	    pack $w.main.f4.f.f1 -padx 2 -pady 2 -fill x -expand yes
	    checkbutton $w.main.f4.f.f1.stencil -text "Use Stencil" \
		-variable $this-use_stencil -command "$this-c needexecute"
	    checkbutton $w.main.f4.f.f1.opacity -text "Outline Levels" \
		-variable $this-outline_levels -command "$this-c needexecute"
	    pack $w.main.f4.f.f1.stencil $w.main.f4.f.f1.opacity -side left
	    
	    frame $w.main.f4.f.f2 -relief flat -borderwidth 2
	    pack $w.main.f4.f.f2 -padx 2 -pady 2 -fill x -expand yes
	    label $w.main.f4.f.f2.l -text "Show level"
	    pack $w.main.f4.f.f2.l -side left
	    frame $w.main.f4.f.f2.f -relief flat -borderwidth 2
	    pack $w.main.f4.f.f2.f -side right -expand yes -fill x
	    set selected 0
	    for { set i 0 } { $i < [set $this-multi_level] } { incr i } {
		checkbutton $w.main.f4.f.f2.f.b$i -text $i \
		    -variable $this-l$i -command "$this-c needexecute" 
		pack $w.main.f4.f.f2.f.b$i -side left
		if { [isOn l$i] } {
		    set selected 1
		}
	    }
	    if { !$selected && [winfo exists $w.main.f4.f.f2.f.b0] } {  
		$w.main.f4.f.f2.f.b0 select 
	    }
	}
    }
    
    method destroy_multi_level { } {
	set w .ui[modname]
	if {[winfo exists $w.main]} {
	    destroy $w.main
	}
	build_ui
    }

    method hasUI {} {
	return [winfo exists .ui[modname]]
    }

    method isOn { bval } {
	return  [set $this-$bval]
    }

}

