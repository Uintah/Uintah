itcl_class PSECommon_Domain_Register {
    inherit Module

    constructor {config} {
	set name Register
	set_defaults
    }

    method set_defaults {} {
    }

    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left
	entry $w.e -textvariable $v
	# bind $w.e <Return> $c
	pack $w.e -side right
    }

    method ui {} {
	set w .ui[modname]
	set n "$this-c needexecute "

	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w 

	### Create the Geometries and Attributes listboxes
	frame $w.main -width 300 -height 200
	pack $w.main -side top -side left
	make_entry $w.main.geom "Geometry:" $this-geom $n
	make_entry $w.main.attrib "Attribute:" $this-attrib $n
	pack $w.main.geom -fill x -padx 5 -pady 5
	pack $w.main.attrib -fill x -padx 5 -pady 5

	button $w.main.go -text "Execute" -relief raised -command $n 
	pack $w.main.go -side left

	pack $w.main -padx 5 -pady 5

	
    }
}
