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
	global $this-head_length
	set $this-head_length 0.3
	global $this-width_scale
	set $this-width_scale 0.1
	global $this-type
	set $this-type 2D
	global $this-exhaustive_flag
	set $this-exhaustive_flag 0
	global $this-skip_node
	set $this-skip_node 1
	$this-c needexecute

	global $this-var_orientation
	set $this-var_orientation 0
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	bind $w.e <Return> $c
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
	wm minsize $w 300 20
	set n "$this-c needexecute "

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2

	make_labeled_radio $w.f.type "NodeHedgehog type:" $n \
		top $this-type \
		{2D 3D}
	pack $w.f.type -side left -padx 5 -anchor w
	
	make_labeled_radio $w.f.shaft "Shaft style:" $n \
		top $this-drawcylinders { {Lines 0} {Cylinders 1} }
	pack $w.f.shaft -side left -padx 5 -anchor w

	frame $w.f.var
	pack $w.f.var -side left -anchor w -pady 2 -ipadx 3

	radiobutton $w.f.var.node -variable $this-var_orientation \
		-command $n -text "Node Centered" -value 0
	pack $w.f.var.node -side top -anchor w -pady 2 -ipadx 3

	radiobutton $w.f.var.cell -variable $this-var_orientation \
		-command $n -text "Cell Centered" -value 1
	pack $w.f.var.cell -side top -anchor w -pady 2 -ipadx 3
	
	radiobutton $w.f.var.face -variable $this-var_orientation \
		-command $n -text "Face Centered" -value 2
	pack $w.f.var.face -side top -anchor w -pady 2 -ipadx 3

	button $w.f.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.f.findxy -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.f.findyz -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.f.findxz -pady 2 -side top -ipadx 3 -anchor e

	frame $w.fskip
	pack $w.fskip -side top -fill x

	global $this-exhaustive_flag
	checkbutton $w.fskip.exh -text "Exhaustive search?" -variable \
		$this-exhaustive_flag
	pack $w.fskip.exh -pady 2 -side right -ipadx 3 -anchor e

	make_entry $w.fskip.skip "Node Skip:" $this-skip_node $n
	pack $w.fskip.skip -side left -padx 5 -anchor w

	frame $w.f2 
	pack $w.f2 -side top -fill x
	expscale $w.f2.length_scale -orient horizontal -label "Length scale:" \
		-variable $this-length_scale -command $n

	pack $w.f2.length_scale -side top -fill x

	scale $w.f2.head_length -orient horizontal -label "Head length:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-head_length -command $n 
	pack $w.f2.head_length -side left -fill x -pady 2
	scale $w.f2.width_scale -orient horizontal -label "Base width:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-width_scale -command $n 
	pack $w.f2.width_scale -side right -fill x -pady 2
	scale $w.f2.shaft_scale -orient horizontal -label "Shaft Radius" \
		-from 0 -to 1 -length 3c \
		-showvalue true -resolution 0.001 \
		-variable $this-shaft_rad -command $n
	pack $w.f2.shaft_scale -side left -fill x -pady 2
	
	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side bottom -expand yes -fill x
    }
}
