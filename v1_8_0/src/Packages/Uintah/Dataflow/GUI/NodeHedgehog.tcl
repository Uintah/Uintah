#  NodeHedgehog.tcl
#  Written by:
#   James Bigler & Steve Parker
#   Department of Computer Science
#   University of Utah
#   June 2000
#  Copyright (C) 2000 SCI Group

itcl_class Uintah_Visualization_NodeHedgehog {
    inherit Module
    constructor {config} {
	set name NodeHedgehog
	set_defaults
    }

    method set_defaults {} {
	global $this-length_scale
	set $this-length_scale 0.1
	global $this-min_crop_length
	set $this-min_crop_length 0
	global $this-max_crop_length
	set $this-max_crop_length 0
	global $this-head_length
	set $this-head_length 0.3
	global $this-width_scale
	set $this-width_scale 0.1
	global $this-type
	set $this-type 2D
	global $this-skip_node
	set $this-skip_node 1

	# max vector stuff
	global $this-max_vector_length
	set $this-max_vector_length 0
	global $this-max_vector_x
	set $this-max_vector_x 0
	global $this-max_vector_y
	set $this-max_vector_y 0
	global $this-max_vector_z
	set $this-max_vector_z 0
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
	pack $w.e -side right
    }
    method make_non_edit_box {w text v} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v -state disabled
	pack $w.e -side right
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 430

	set n "$this-c needexecute "
	
	frame $w.f1
	pack $w.f1 -side top -fill x -padx 2 -pady 2

	make_labeled_radio $w.f1.type "NodeHedgehog type:" $n \
		top $this-type \
		{2D 3D}
	pack $w.f1.type -side left -padx 2 -anchor w
	
	make_labeled_radio $w.f1.shaft "Shaft style:" $n \
		top $this-drawcylinders { {Lines 0} {Cylinders 1} }
	pack $w.f1.shaft -side left -padx 2 -anchor w

	frame $w.f1.b
	pack $w.f1.b -side right -fill x -padx 2 -pady 2

	button $w.f1.b.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.f1.b.findxy -pady 2 -side top -anchor e -fill x
	
	button $w.f1.b.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.f1.b.findyz -pady 2 -side top -anchor e -fill x
	
	button $w.f1.b.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.f1.b.findxz -pady 2 -side top -anchor e -fill x

	frame $w.f2
	pack $w.f2 -side top -fill x -padx 2 -pady 2

	frame $w.f2.var
	pack $w.f2.var -side left -anchor w -pady 2 -ipadx 3

	checkbutton $w.f2.var.normhead -text "Uniform Headsize" -command $n \
		-variable $this-norm_head
	pack $w.f2.var.normhead -side left -anchor w -pady 2 -ipadx 3

	make_entry $w.f2.var.skip "Node Skip:" $this-skip_node $n
	pack $w.f2.var.skip -side left -padx 5 -pady 2 -anchor w -fill x

	frame $w.max -relief sunken -borderwidth 1
	pack $w.max -side top -padx 8 -ipady 2 -pady 3 -fill x
	
	make_non_edit_box $w.max.length "Max length (non-cropped):" \
		$this-max_vector_length
	pack $w.max.length -side top -padx 5 -pady 2 -anchor w -fill x

	make_non_edit_box $w.max.x "Max.x:" $this-max_vector_x
	pack $w.max.x -side top -padx 5 -pady 2 -anchor w -fill x

	make_non_edit_box $w.max.y "Max.y:" $this-max_vector_y
	pack $w.max.y -side top -padx 5 -pady 2 -anchor w -fill x

	make_non_edit_box $w.max.z "Max.z:" $this-max_vector_z
	pack $w.max.z -side top -padx 5 -pady 2 -anchor w -fill x

	frame $w.f5 
	pack $w.f5 -side top -fill x
	expscale $w.f5.length_scale -orient horizontal -label "Length scale:" \
		-variable $this-length_scale -command $n

	pack $w.f5.length_scale -side top -fill x

	expscale $w.f5.min_length -orient horizontal \
		-label "Minimum Length:" \
		-variable $this-min_crop_length -command $n

	pack $w.f5.min_length -side top -fill x

	expscale $w.f5.max_length -orient horizontal \
		-label "Maximum Length (not used if equal to 0):" \
		-variable $this-max_crop_length -command $n

	pack $w.f5.max_length -side top -fill x

	scale $w.f5.head_length -orient horizontal -label "Head length:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-head_length -command $n 
	pack $w.f5.head_length -side left -fill x -pady 2
	scale $w.f5.width_scale -orient horizontal -label "Base width:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-width_scale -command $n 
	pack $w.f5.width_scale -side right -fill x -pady 2
	scale $w.f5.shaft_scale -orient horizontal -label "Shaft Radius" \
		-from 0 -to 1 -length 3c \
		-showvalue true -resolution 0.001 \
		-variable $this-shaft_rad -command $n
	pack $w.f5.shaft_scale -side left -fill x -pady 2
	
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}
