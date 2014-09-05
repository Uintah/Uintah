
catch {rename ParticleVis ""}

itcl_class Uintah_Visualization_ParticleVis {

    inherit Module
    protected r ""
    protected l_s ""
    protected changed 0

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
	global $this-auto_radius
	global $this-polygons
	global $this-show_nth
	global $this-drawVectors
	global $this-length_scale
	global $this-auto_length_scale
	global $this-head_length
	global $this-shaft_rad
	global $this-width_scale
	global $this-drawcylinders
	global $this-drawspheres
	global $this-isFixed
	global $this-min_
	global $this-max_
	global $this-min_crop_length
	global $this-max_crop_length
	set $this-current_time 0
	set $this-radius 0.01
	set $this-auto_radius 1
	set $this-polygons 32
	set $this-show_nth 1
	set $this-drawVectors 0
	set $this-length_scale 0.1
	set $this-auto_length_scale 1
	set $this-head_length 0.3
	set $this-shaft_rad 0.1
	set $this-width_scale 0.1
	set $this-drawcylinders 0
	set $this-drawspheres 0
	set $this-isFixed 0
	set $this-min_ 0
	set $this-max_ 1
	set $this-min_crop_length 0
	set $this-max_crop_length 0

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

	frame $w.f1.rad -relief flat
	pack $w.f1.rad -side top -fill x
	checkbutton $w.f1.rad.auto_radius -text "auto radius" \
	    -variable $this-auto_radius -command $n
	set r [expscale $w.f1.rad.radius -label "Radius:" -orient horizontal \
		    -variable $this-radius -command $n ]
	pack $w.f1.rad.radius $w.f1.rad.auto_radius -side bottom -fill x

	scale $w.f1.res -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command $n \
	    -from 8 -to 400 -tickinterval 392

	pack $w.f1.res -side top -expand yes -fill x

	scale $w.f1.nth -label "Show Nth Particle:" -orient horizontal \
	    -variable $this-show_nth -command $n \
	    -from 1 -to 100 -tickinterval 99 -resolution 1
	
	pack $w.f1.nth -side top -expand yes -fill x

	frame $w.f3 -relief groove -borderwidth 2
	pack $w.f3 -side top -expand yes -fill both
	label $w.f3.l -text "Particle Scale Control"
	pack $w.f3.l -side top -expand yes -fill x
	frame $w.f3.sf -relief flat -borderwidth 2
	pack $w.f3.sf -side top -expand yes -fill x
	set sf $w.f3.sf
	checkbutton $sf.cb -text "Fixed Range" -variable $this-isFixed \
	    -onvalue 1 -offvalue 0 -command "$this fixedScale"
	label $sf.l1 -text "min: "
	entry $sf.e1 -textvariable $this-min_
	label $sf.l2 -text " max: "
	entry $sf.e2 -textvariable $this-max_
	pack $sf.cb $sf.l1 $sf.e1 $sf.l2 $sf.e2 -side left \
	    -expand yes -fill x -padx 2 -pady 2

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

	frame $w.f2.ls -relief flat
	pack $w.f2.ls -side top -fill x
	checkbutton $w.f2.ls.auto_length_scale -text "auto length scale" \
	    -variable $this-auto_length_scale -command $n
	expscale $w.f2.ls.length_scale -label "Length scale:" \
	    -orient horizontal  -variable $this-length_scale \
	    -command $n 
	pack $w.f2.ls.length_scale $w.f2.ls.auto_length_scale -side bottom -fill x
	expscale $w.f2.min_length -orient horizontal \
		-label "Minimum Length:" \
		-variable $this-min_crop_length -command $n

	pack $w.f2.min_length -side top -fill x

	expscale $w.f2.max_length -orient horizontal \
		-label "Maximum Length (not used if equal to 0):" \
		-variable $this-max_crop_length -command $n

	pack $w.f2.max_length -side top -fill x


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
	
	$this scalable 0
    }

    method scalable { truefalse } {
	set w .ui[modname]
	if {[winfo exists $w]} { 
	    puts "scalable is $truefalse in tcl"
	    if { $truefalse == 1} {
		$w.f3.sf.cb configure -state normal -foreground black
		$w.f3.l configure -foreground black
	    } else {
		set color "#505050" 
		$w.f3.sf.cb configure -state disabled -foreground $color
		$w.f3.l configure  -foreground $color
	    }
	    $this fixedScale
	}
    }
    
    method fixedScale { } {
	global $this-isFixed
	set w .ui[modname]

	if {[set $this-isFixed] == 1 } {
	    $w.f3.sf.l1 configure -foreground black
	    $w.f3.sf.e1 configure -state normal -foreground black
	    $w.f3.sf.l2 configure -foreground black
	    $w.f3.sf.e2 configure -state normal -foreground black
	} else {
	    set color "#505050"
	    
	    $w.f3.sf.l1 configure -foreground $color
	    $w.f3.sf.e1 configure -state disabled -foreground $color
	    $w.f3.sf.l2 configure -foreground $color
	    $w.f3.sf.e2 configure -state disabled -foreground $color
	}


	if { $changed != [set $this-isFixed] } {
	    set changed [set $this-isFixed]
	    $this-c needexecute
	}
	
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

