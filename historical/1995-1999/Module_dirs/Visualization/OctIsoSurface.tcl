itcl_class OctIsoSurface {
    inherit Module
    constructor {config} {
        set name OctIsoSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-isoval
	global $this-isoval_from
	global $this-isoval_to
	global $this-levels
	global $this-depth
	set $this-isoval 1
	set $this-isoval_from 0
	set $this-isoval_to 100
	set $this-levels 5
	set $this-depth 1
    }
    method ui {} {
        set w .ui$this
        if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w
        wm minsize $w 300 100
        frame $w.f -width 400 -height 500
        pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
        set n "$this-c needexecute"
        frame $w.f.iso
        pack $w.f.iso -side top 
        label $w.f.iso.label -text "Octree IsoSurfacing"
	global $this-isoval_from
	global $this-isoval_to
	scale $w.f.iso.isoval -from [set $this-isoval_from] \
		-to [set $this-isoval_to] -label "Isovalue: "\
		-showvalue true -resolution 0 -digits 3 -resolution 0.01 \
		-variable $this-isoval -orient horizontal -length 250 \
		-command $n
	trace variable $this-isoval_from w "$this change_isoval_from"
	trace variable $this-isoval_to w "$this change_isoval_to"
	global $this-levels
	scale $w.f.iso.depth -from 1 -to [set $this-levels] -label "Depth: "\
		-showvalue true \
		-variable $this-depth -orient horizontal -length 250 \
		-command $n
	trace variable $this-levels w "$this change_levels"
        checkbutton $w.f.iso.same -command $n \
                -text "Same Input" -variable $this-same_input
        $w.f.iso.same select
        button $w.f.iso.execute -command $n -text "Execute"
        pack $w.f.iso.label $w.f.iso.isoval $w.f.iso.depth \
		$w.f.iso.same $w.f.iso.execute -side top
    }

    method change_levels {n1 n2 op} {
	global $this-levels
	global .ui$this.f.iso.depth
	.ui$this.f.iso.depth configure -to [set $this-levels]
    }

    method change_isoval_from {n1 n2 op} {
	global $this-isoval_from
	global .ui$this.f.iso.isoval
	.ui$this.f.iso.isoval configure -from [set $this-isoval_from]
    }
	
    method change_isoval_to {n1 n2 op} {
	global $this-isoval_to
	global .ui$this.f.iso.isoval
	.ui$this.f.iso.isoval configure -to [set $this-isoval_to]
    }
	
}
