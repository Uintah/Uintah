
itcl_class MorphMesher3d {
    inherit Module
    constructor {config} {
	set name MorphMesher3d
	set_defaults
    }
    method set_defaults {} {
	set $this-num_layers 4
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2

	frame $w.f.num_layers
	pack $w.f.num_layers
	label $w.f.num_layers.label -text "Number of mesh layers: "
	scale $w.f.num_layers.scale -variable $this-num_layers \
		-from 1 -to 9 -command "$this-c needexecute "\
		-showvalue true -tickinterval 1 \
		-orient horizontal
	pack $w.f.num_layers.label $w.f.num_layers.scale -fill x
    }
}

