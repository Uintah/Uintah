
catch {rename MeshView ""}

itcl_class MeshView {
    inherit Module
    constructor {config} {
	set name MeshView
	set_defaults
    }
    method set_defaults {} {
	global $this-allLevels
	set $this-allLevels 1
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
		raise $w
		return;
	}

	toplevel $w
	wm minsize $w 200 20
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	scale $w.f.numLevels -variable $this-numLevels -from 0 -to 10 \
		-label "Number of Levels" -command $n \
		-orient horizontal
	pack $w.f.numLevels -side top -fill x

	scale $w.f.seedTet -variable $this-seedTet -from 0 -to 1 -command $n \
		-orient horizontal -label "Starting Tetrahedron"
	pack $w.f.seedTet -side top -fill x

	
	scale $w.f.clipX -variable $this-clipX -from -1 -to 1 \
		-orient horizontal -command $n -label X
	pack $w.f.clipX -side top -fill x

	scale $w.f.clipNX -variable $this-clipNX -from -1 -to 1 \
		-orient horizontal -command $n -label "Negative X"
	pack $w.f.clipNX -side top -fill x

	scale $w.f.clipY -variable $this-clipY -from -1 -to 1 \
		-orient horizontal -command $n -label Y
	pack $w.f.clipY -side top -fill x

	scale $w.f.clipNY -variable $this-clipNY -from -1 -to 1 \
		-orient horizontal -command $n -label "Negative Y"
	pack $w.f.clipNY -side top -fill x

	scale $w.f.clipZ -variable $this-clipZ -from -1 -to 1 \
		-orient horizontal -command $n -label Z
	pack $w.f.clipZ -side top -fill x

	scale $w.f.clipNZ -variable $this-clipNZ -from -1 -to 1 \
		-orient horizontal -command $n -label "Negative Z"
	pack $w.f.clipNZ -side top -fill x

	frame $w.f.levels
	pack $w.f.levels -side left
	label $w.f.levels.label -text "Show all levels?"
	radiobutton $w.f.levels.on -text No -relief flat \
		-variable $this-allLevels -value 0  -command $n
	radiobutton $w.f.levels.off -text Yes -relief flat \
		-variable $this-allLevels -value 1  -command $n
	pack $w.f.levels.label -side left
	pack $w.f.levels.on -side left
	pack $w.f.levels.off -side left
    }
    method set_minmax_nl {min max} {
	set w .ui$this
	$w.f.numLevels configure -from $min -to $max
    }
    method set_minmax_numTet {min max} {
	set w .ui$this
	$w.f.seedTet configure -from $min -to $max
    }
    method set_bounds {xmin xmax ymin ymax zmin zmax} {
	set w .ui$this
	$w.f.clipX configure -from $xmin -to $xmax
	$w.f.clipNX configure -from $xmin -to $xmax
	$w.f.clipY configure -from $ymin -to $ymax
	$w.f.clipNY configure -from $ymin -to $ymax
	$w.f.clipZ configure -from $zmin -to $zmax
	$w.f.clipNZ configure -from $zmin -to $zmax
    }
}


