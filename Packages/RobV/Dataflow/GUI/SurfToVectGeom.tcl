#  SurfToVectGeom.tcl
#  Written by:
#   Robert Van Uitert
#   Department of Computer Science
#   University of Utah
#   July 1999
#  Copyright (C) 1999 SCI Group

catch {rename RobV_MEG_SurfToVectGeom ""}

itcl_class RobV_MEG_SurfToVectGeom {
    inherit Module
    constructor {config} {
	set name SurfToVectGeom
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
	set $this-type 3D
	global $this-exhaustive_flag
	set $this-exhaustive_flag 0
	global $this-max_vect
	set $this-max_vect 10000
    }

    method ui {} {
	set w .ui[modname]
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 40
	set n "$this-c needexecute "

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2

	
	make_labeled_radio $w.f.shaft "Shaft style:" $n \
		top $this-drawcylinders { {Lines 0} {Cylinders 1} }
	pack $w.f.shaft -side left -padx 5 -anchor w
	
	global $this-exhaustive_flag
	checkbutton $w.f.exh -text "Exhaustive search?" -variable \
		$this-exhaustive_flag
	pack $w.f.exh -pady 2 -side top -ipadx 3 -anchor e

	expscale $w.length_scale -orient horizontal -label "Length scale:" \
		-variable $this-length_scale -command $n
	pack $w.length_scale -side top -fill x

	scale $w.head_length -orient horizontal -label "Head length:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-head_length -command $n 
	pack $w.head_length -side left -fill x -pady 2
	scale $w.width_scale -orient horizontal -label "Base width:" \
		-from 0 -to 1 -length 3c \
                -showvalue true \
                -resolution 0.001 \
		-variable $this-width_scale -command $n 
	pack $w.width_scale -side right -fill x -pady 2
	scale $w.shaft_scale -orient horizontal -label "Shaft Radius" \
		-from 0 -to 1 -length 3c \
		-showvalue true -resolution 0.001 \
		-variable $this-shaft_rad -command $n
	pack $w.shaft_scale -side left -fill x -pady 2

	scale $w.max_vect -orient horizontal -label "Max Vector Length:" \
		-from 0 -to 10000 -length 5c \
		-showvalue true -resolution 1 \
		-variable $this-max_vect -command $n
	pack $w.max_vect -side bottom -fill x

    }
}







