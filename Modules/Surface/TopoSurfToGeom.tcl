itcl_class TopoSurfToGeom {
    inherit Module
    constructor {config} {
        set name TopoSurfToGeom
        set_defaults
    }

    method set_defaults {} {
	global $this-mode
	set $this-mode patches
    }

    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2
	make_labeled_radio $w.mode "Viz Mode" ""\
                top $this-mode \
                {{Patches patches}  \
		{Wires wires} \
		{Junctions junctions}}

	pack $w.mode -fill x
    }
}
