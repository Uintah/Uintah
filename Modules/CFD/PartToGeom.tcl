
catch {rename PartToGeom ""}

itcl_class PartToGeom {
    inherit Module
    constructor {config} {
	set name PartToGeom
	set_defaults
    }
    method set_defaults {} {
	global $this-current_time
	set $this-current_time 0

    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    wm deiconify $w
	    raise $w
	    return;
	}
	set type ""
  
	toplevel $w
	wm minsize $w 300 20

	expscale $w.time -label "Time:" -orient horizontal \
		-variable $this-current_time -command "$this-c needexecute"
	pack $w.time -side top -fill x

	make_labeled_radio $w.geom "Particle Geometry" "$this-c needexecute" \
		top $this-drawspheres { {Points 0} {Spheres 1} }
	pack $w.geom -side top -fill x

	expscale $w.radius -label "Radius:" -orient horizontal \
		-variable $this-radius -command "$this-c needexecute"
	pack $w.radius -side top -fill x

	scale $w.res -label "Polygons:" -orient horizontal \
	    -variable $this-polygons -command "$this-c needexecute" \
	    -from 8 -to 400 -tickinterval 392

	pack $w.res -side top -expand yes -fill x
    }
}

