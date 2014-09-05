#
#  Hedgehog.tcl
#
#  Written by:
#   Colette Mullenhoff
#   Department of Computer Science
#   University of Utah
#   May. 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class PSECommon_Visualization_HedgehogLitLines {
    inherit Module
    protected l_s ""
    constructor {config} {
	set name Hedgehog
	set_defaults
    }
    destructor {
	set l_s ""
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
	global $this-use_locus
	set $this-use_locus 0
	$this-c needexecute
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

	make_labeled_radio $w.f.type "Hedgehog type:" "$this exec_check2" \
		top $this-type \
		{2D 3D}
	pack $w.f.type -side left -padx 5 -anchor w

	make_labeled_radio $w.f.shaft "Shaft style:" $n \
		top $this-drawcylinders { {Lines 0} {Cylinders 1} }
	pack $w.f.shaft -side left -padx 5 -anchor w
	
	button $w.f.findxy -text "Find XY" -command "$this-c findxy"
	pack $w.f.findxy -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findyz -text "Find YZ" -command "$this-c findyz"
	pack $w.f.findyz -pady 2 -side top -ipadx 3 -anchor e
	
	button $w.f.findxz -text "Find XZ" -command "$this-c findxz"
	pack $w.f.findxz -pady 2 -side top -ipadx 3 -anchor e

#	expscale $w.length_scale -orient horizontal -label "Length scale:" \
#		-variable $this-length_scale -command "$this exec_check"
#	pack $w.length_scale -side top -fill x

	checkbutton $w.use_loc -text "Use locus?" -variable $this-use_locus \
		-command $n
	pack $w.use_loc -side top

	frame $w.a
	scale $w.locus_size -orient horizontal \
		-label "Percent of volume in locus:" \
		-from 0 -to 100 -showvalue true -resolution 1 \
		-variable $this-locus_size
	pack $w.locus_size -side top -fill x -pady 2

	set l_s [ expscale $w.a.length_scale -orient horizontal \
		    -label "Length scale:" \
		    -variable $this-length_scale -command $n ]
	pack $w.a.length_scale -side top -fill x
	scale $w.a.head_length -orient horizontal -label "Head length:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-head_length -command "$this exec_check"
	pack $w.a.head_length -side left -fill x -pady 2
	scale $w.a.width_scale -orient horizontal -label "Base width:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-width_scale -command "$this exec_check"
	pack $w.a.width_scale -side right -fill x -pady 2
	pack $w.a -side top -fill x -pady 2
	scale $w.a.shaft_scale -orient horizontal -label "Shaft Radius" \
		-from 0 -to 1 -length 3c \
		-showvalue true -resolution 0.001 \
		-variable $this-shaft_rad -command $n
	pack $w.a.shaft_scale -side left -fill x -pady 2


	button $w.close -text "Close" -command "wm withdraw $w"
	pack $w.close -side top -expand yes -fill x

    }


    method exec_check2 { } {
	exec_check 0
    }
 
    method exec_check { v } {
	global $this-use_locus
	if {[set $this-use_locus]} return
	$this-c needexecute
    }
}

