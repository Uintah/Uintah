
catch {rename PartToGeom ""}

itcl_class PartToGeom {

    inherit Module

    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }

    constructor {config} {
	set name PartToGeom
	set_defaults
    }

    destructor {
	puts "destructor:  PartToGeom"
	set w .ui[modname]
	if {[winfo exists $w]} {
	    ::delete class expscale 
	}
	puts "done"
    }

    method set_defaults {} {
	global $this-current_time
	set $this-current_time 0
    }

    method ui {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	set type ""
  
	global $this-length_scale
	global $this-head_length
	global $this-width_scale
	global $this-drawcylinders
	global $this-shaft_rad

	set n "$this-c needexecute"

	toplevel $w
	wm minsize $w 300 20
	frame $w.f1 -relief groove -borderwidth 2
	pack $w.f1 -side top -expand yes -fill both

	expscale $w.f1.time -label "Time:" -orient horizontal \
	  -variable $this-current_time -command $n
	pack $w.f1.time -side top -fill x

	make_labeled_radio $w.f1.geom "Particle Geometry" $n \
	     top $this-drawspheres { {Points 0} {"Spheres (picking mode)" 1} }
	pack $w.f1.geom -side top -fill x

	expscale $w.f1.radius -label "Radius:" -orient horizontal \
		-variable $this-radius -command $n
	pack $w.f1.radius -side top -fill x

	scale $w.f1.res -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392

	pack $w.f1.res -side top -expand yes -fill x


	frame $w.f2 -relief groove -borderwidth 2
	pack $w.f2 -side top -expand yes -fill both
	label $w.f2.l -text "Vector Controls\n"
	pack $w.f2.l -side top -expand yes -fill x

	frame $w.f2.f -relief flat
	pack $w.f2.f -side top -expand yes -fill x
	checkbutton $w.f2.f.chk -text "Show Vectors" -variable $this-drawVectors \
	    -command $n -offvalue 0 -onvalue 1
	pack $w.f2.f.chk -side left -anchor w -expand yes -fill x

	make_labeled_radio $w.f2.f.shaft "Shaft style:" $n \
		top $this-drawcylinders { {Lines 0} {Cylinders 1} }
	pack $w.f2.f.shaft -side left -padx 5 -anchor w

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


	button $w.close -text "Close" -command "$this close"
	pack $w.close -side top -expand yes -fill x

    }

    method close {} {
	puts "closing PartToGeom Ui"
	set w .ui[modname]
	::delete class expscale
	destroy $w
    }
}

