
itcl_class Uintah_MPMViz_ParticleViz {

    inherit Module

    constructor {config} {
	set name ParticleViz
	set_defaults
    }

    destructor {
    }

    method set_defaults {} {
	global $this-current_time
	set $this-current_time 0

	global $this-track_latest
	set $this-track_latest 1

	global $this-type
	set $this-type "Points"

	global $this-radius
	set $this-radius 1.0
    }
    public active false

    method ui {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
  
	global $this-current_time
	global $this-track_latest
	global $this-type
	global $this-radius


	toplevel $w
	wm minsize $w 400 20
	set minmax [$this-c getMinMax]
	set min [lindex $minmax 0]
	set max [lindex $minmax 1]
	scale $w.current_time -label "Time:" -orient horizontal \
		-variable $this-current_time -command "$this changetime" \
		-from $min -to $max -resolution 0
	frame $w.f
	label $w.f.label -text "Track latest"
	checkbutton $w.f.track_latest -variable $this-track_latest 
	pack $w.f.track_latest -side right
	pack $w.f.label -side right -padx 2 -anchor e
	make_labeled_radio $w.type "Draw as:" "$this updateCur" left \
		$this-type {{Points} {Spheres}}

	set r [expscale $w.radius -label "Radius:" -orient horizontal \
		    -variable $this-radius -command "$this update" ]
	scale $w.nu -label "nu:" -orient horizontal -variable $this-nu \
		-command "$this update" -from 3 -to 20
	scale $w.nv -label "nv:" -orient horizontal -variable $this-nv \
		-command "$this update" -from 2 -to 20

	pack $w.current_time $w.f $w.type $w.radius -side top -fill x
	pack $w.nu $w.nv -side left -fill x -expand yes
    }

    method updateMinMax {min max} {
	set w .ui[modname]
	puts "max is $max"
	$w.current_time configure -from $min -to $max
	if {[set $this-track_latest]} {
	    set $this-current_time $max
	}
	set active true
    }
    method updateMax {max} {
	set w .ui[modname]
	$w.current_time configure -to $max
	if {[set $this-track_latest]} {
	    set $this-current_time $max
	    $this-c update $max
	}
    }
    method update {junk} {
	global $this-current_time
	$this-c update [set $this-current_time]
    }
    method updateCur {} {
	global $this-current_time
	$this-c update [set $this-current_time]
    }
    method changetime {curtime} {
	if {$active} {
	    set $this-track_latest 0
	}
	$this-c update $curtime
    }
}
