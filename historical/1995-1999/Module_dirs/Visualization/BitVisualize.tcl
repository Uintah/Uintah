
itcl_class BitVisualize {
    inherit Module
    constructor {config} {
	set name BitVisualize
	set_defaults
    }
    method set_defaults {} {
	global $this-emit_surface
	set $this-emit_surface 0
	global $this-vol_render
	set $this-vol_render 0
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
	set n "$this-c needexecute "
	
	frame $w.f.materials
	pack $w.f.materials -side top

	label $w.f.materials.label -text "Materials: "
	checkbutton $w.f.materials.v1 -text "Skin" -relief flat \
		-variable $this-iso1
	set $this-iso1 1
	checkbutton $w.f.materials.v2 -text "Bone" -relief flat \
		-variable $this-iso2
	set $this-iso2 1
	checkbutton $w.f.materials.v3 -text "Fluid" -relief flat \
		-variable $this-iso3
	set $this-iso3 1
	checkbutton $w.f.materials.v4 -text "Grey Matter" -relief flat \
		-variable $this-iso4
	set $this-iso4 1
	checkbutton $w.f.materials.v5 -text "White Matter" -relief flat \
		-variable $this-iso5
	set $this-iso5 1
	pack $w.f.materials.label $w.f.materials.v1 \
		$w.f.materials.v2  $w.f.materials.v3 \
		$w.f.materials.v4 $w.f.materials.v5 -side left
	
	checkbutton $w.f.emit_surface -text "Emitting Surface" -relief flat \
	    -variable $this-emit_surface
	button $w.f.execute -text "Execute" -relief raised -command $n
	pack $w.f.emit_surface $w.f.execute -side top
    }
}
