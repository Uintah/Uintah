itcl_class FieldMedianFilter {
    inherit Module
    constructor {config} {
	set name FieldMedianFilter
	set_defaults
    }
    method set_defaults {} {
	global $this-kernel
	set $this-kernel 2
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 200 50
	frame $w.f
	global $this-kernel
	scale $w.f.s -label "Kernel size (voxels): " -variable $this-kernel \
		-orient horizontal -from 0 -to 32 -showvalue true
	pack $w.f.s -fill both -expand 1
	pack $w.f -fill both -expand 1
    }
}
