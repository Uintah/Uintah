
catch {rename VectorFieldOceanReader ""}

itcl_class VectorFieldOceanReader {
    inherit Module
    constructor {config} {
	set name VectorFieldOceanReader
	set_defaults
    }
    method set_defaults {} {
	global $this-downsample
	set $this-downsample 4
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	makeFilebox $w.f $this-filename "$this-c needexecute" "destroy $w"
	pack $w.f -side top
	make_labeled_radio $w.downsample "Surface downsample: " "$this-c needexecute" left $this-downsample \
	    {1 2 4 8 16}
	pack $w.downsample -side bottom
    }
}

