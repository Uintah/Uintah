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


catch {rename GridVolVis ""}

itcl_class Kurt_Visualization_GridVolVis {
    inherit Module
    constructor {config} {
	set name GridVolVis
	set_defaults
    }
    method set_defaults {} {
	global $this-draw_mode
	set $this-draw_mode 0
	global $this-num_slices
	set $this-num_slices 64
	global $this-alpha_scale
	set $this-alpha_scale 0
	global $this-render_style
	set $this-render_style 0
	global $this-interp_mode 
	set $this-interp_mode 1
	global $this-contrast
	set $this-contrast 0.5
	global $this-contrastfp
	set $this-contrastfp 0.5
	global $this-max_brick_dim_
	global $this-min_
	global $this-max_
	global $this-is_fixed_
	set $this-max_brick_dim_ 0
	set $this-min_ 0
	set $this-max_ 1
	set $this-is_fixed_ 0
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
	label $w.f.l -text "Rendering Style"
	radiobutton $w.f.modeo -text "Over Operator" -relief flat \
		-variable $this-render_style -value 0 \
		-anchor w -command $n

	radiobutton $w.f.modem -text "MIP" -relief flat \
		-variable $this-render_style -value 1 \
		-anchor w -command $n

	radiobutton $w.f.modea -text "Attenuate" -relief flat \
		-variable $this-render_style -value 2 \
		-anchor w -command $n

	pack $w.f.l $w.f.modeo $w.f.modem $w.f.modea \
		-side top -fill x

	frame $w.frame3 -relief groove -borderwidth 2
	pack $w.frame3 -padx 2 -pady 2 -fill x

	label $w.frame3.l -text "Interpolation Mode"
	radiobutton $w.frame3.interp -text "Interpolate" -relief flat \
		-variable $this-interp_mode -value 1 \
		-anchor w -command $n

	radiobutton $w.frame3.near -text "Nearest" -relief flat \
		-variable $this-interp_mode -value 0 \
		-anchor w -command $n

	pack $w.frame3.l $w.frame3.interp $w.frame3.near \
		-side top -fill x

	global $this-num_slices
	scale $w.nslice -variable $this-num_slices \
		-from 64 -to 1024 -label "Number of Slices" \
		-showvalue true \
		-orient horizontal \


	global $this-alpha_scale
	
	scale $w.stransp -variable $this-alpha_scale \
		-from -1.0 -to 1.0 -label "Slice Transparency" \
		-showvalue true -resolution 0.001 \
		-orient horizontal 

	pack $w.stransp $w.nslice  -side top -fill x

	button $w.exec -text "Execute" -command $n
	pack $w.exec -side top -fill x
	bind $w.nslice <ButtonRelease> $n
	bind $w.stransp <ButtonRelease> $n
	
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
