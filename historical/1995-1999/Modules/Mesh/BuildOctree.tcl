itcl_class BuildOctree {
    inherit Module
    constructor {config} {
	set name BuildOctree
	set_defaults
    }
    method set_defaults {} {
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
	frame $w.f.mesh
	pack $w.f.mesh -side top 
	label $w.f.mesh.label -text "Octree Discretization"
	frame $w.f.mesh.refine
	frame $w.f.mesh.decimate
	pack $w.f.mesh.label $w.f.mesh.refine $w.f.mesh.decimate \
		-side top
	label $w.f.mesh.refine.label -text "Refine..."
	label $w.f.mesh.decimate.label -text "Decimate..."
	button $w.f.mesh.refine.full -command "$this-c bottom_level" -text \
		"Completely"
	button $w.f.mesh.refine.gl -command "$this-c push_all_levels" -text \
		"Globally"
	button $w.f.mesh.refine.lcl -command "$this-c push_level" -text \
		"Locally"
	button $w.f.mesh.decimate.full -command "$this-c top_level" -text \
		"Completely"
	button $w.f.mesh.decimate.gl -command "$this-c pop_all_levels" -text \
		"Globally"
	button $w.f.mesh.decimate.lcl -command "$this-c pop_level" -text \
		"Locally"
	pack $w.f.mesh.refine.label $w.f.mesh.refine.full $w.f.mesh.refine.gl \
		$w.f.mesh.refine.lcl -side left -fill x -expand yes
	pack $w.f.mesh.decimate.label $w.f.mesh.decimate.full \
		$w.f.mesh.decimate.gl $w.f.mesh.decimate.lcl \
		-side left -fill x -expand yes
	frame $w.f.exe
	pack $w.f.exe -side top
	checkbutton $w.f.exe.same -command "$this maybe_execute" \
		-text "Same Input" -variable $this-same_input
	$w.f.exe.same select
	button $w.f.exe.execute -command $n -text "Execute"
	pack $w.f.exe.same $w.f.exe.execute -side top
    }

    method maybe_execute {} {
	global $this-same_input
	if {[set $this-same_input]} {
	    $this-c needexecute
	}
    }
}
