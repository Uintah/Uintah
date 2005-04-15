
catch {rename ParticleVis ""}

itcl_class Uintah_Visualization_ParticleVis {

    inherit Module
    protected r ""
    protected l_s ""

    constructor {config} {
	set name ParticleVis
	set_defaults
    }

    destructor {
	puts "destructor:  ParticleVis"
	set w .ui[modname]
	if {[winfo exists $w]} {
	    ::delete object $r
	    ::delete object $l_s
       }
	set r ""
	set l_s ""
	puts "done"
    }

    method set_defaults {} {
	global $this-current_time
	global $this-radius
	global $this-polygons
	global $this-show_nth
	global $this-drawVectors
	global $this-length_scale
	global $this-head_length
	global $this-shaft_rad
	global $this-width_scale
	global $this-drawcylinders
	global $this-drawspheres
	set $this-current_time 0
	set $this-radius 0.01
	set $this-polygons 32
	set $this-show_nth 2
	set $this-drawVectors 0
	set $this-length_scale 0.1
	set $this-head_length 0.3
	set $this-shaft_rad 0.1
	set $this-width_scale 0.1
	set $this-drawcylinders 0
	set $this-drawspheres 0
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

	make_labeled_radio $w.f1.geom "Particle Geometry" $n \
	     top $this-drawspheres { {Points 0} {"Spheres (picking mode)" 1} }
	pack $w.f1.geom -side top -fill x

	set r [expscale $w.f1.radius -label "Radius:" -orient horizontal \
		    -variable $this-radius -command $n ]
	pack $w.f1.radius -side top -fill x

	scale $w.f1.res -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392

	pack $w.f1.res -side top -expand yes -fill x

	scale $w.f1.nth -label "Show Nth Particle:" -orient horizontal \
	    -variable $this-show_nth -command $n \
	    -from 1 -to 100 -tickinterval 99 -resolution 1
	
	pack $w.f1.nth -side top -expand yes -fill x

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

	set l_s [expscale $w.f2.length_scale -label "Length scale:" \
		     -orient horizontal  -variable $this-length_scale \
		     -command $n ]
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
	pack $w.close -side top -expand yes -fill x

    }

    method close {} {
	puts "closing ParticleVis Ui"
	set w .ui[modname]
	
	::delete object $t
	::delete object $r
	::delete object $l_s
	destroy $w
    }
}

