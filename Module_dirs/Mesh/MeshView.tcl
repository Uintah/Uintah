
proc uiMeshView {modid} {
	set w .ui$modid
	if {[winfo exists $w]} {
		raise $w
		return;
	}

	toplevel $w
	wm minsize $w 200 20
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$modid needexecute "

	global numLevels,$modid
	scale $w.f.numLevels -variable numLevels,$modid -from 0 -to 10 \
		-label "Number of Levels" -command $n \
		-orient horizontal
	pack $w.f.numLevels -side top -fill x

	global seedTet,$modid
	scale $w.f.seedTet -variable seedTet,$modid -from 0 -to 1 -command $n \
		-orient horizontal -label "Starting Tetrahedron"
	pack $w.f.seedTet -side top -fill x

	
	global clipX,$modid
	scale $w.f.clipX -variable clipX,$modid -from -1 -to 1 \
		-orient horizontal -command $n -label X
	pack $w.f.clipX -side top -fill x

	global clipNX,$modid
	scale $w.f.clipNX -variable clipNX,$modid -from -1 -to 1 \
		-orient horizontal -command $n -label "Negative X"
	pack $w.f.clipNX -side top -fill x

	global clipY,$modid
	scale $w.f.clipY -variable clipY,$modid -from -1 -to 1 \
		-orient horizontal -command $n -label Y
	pack $w.f.clipY -side top -fill x

	global clipNY,$modid
	scale $w.f.clipNY -variable clipNY,$modid -from -1 -to 1 \
		-orient horizontal -command $n -label "Negative Y"
	pack $w.f.clipNY -side top -fill x

	global clipZ,$modid
	scale $w.f.clipZ -variable clipZ,$modid -from -1 -to 1 \
		-orient horizontal -command $n -label Z
	pack $w.f.clipZ -side top -fill x

	global clipNZ,$modid
	scale $w.f.clipNZ -variable clipNZ,$modid -from -1 -to 1 \
		-orient horizontal -command $n -label "Negative Z"
	pack $w.f.clipNZ -side top -fill x

	global allLevels,$modid
	set allLevels,$modid 1
	frame $w.f.levels
	pack $w.f.levels -side left
	label $w.f.levels.label -text "Show all levels?"
	radiobutton $w.f.levels.on -text No -relief flat \
		-variable allLevels,$modid -value 0  -command $n
	radiobutton $w.f.levels.off -text Yes -relief flat \
		-variable allLevels,$modid -value 1  -command $n
	pack $w.f.levels.label -side left
	pack $w.f.levels.on -side left
	pack $w.f.levels.off -side left
}

proc MeshView_set_minmax_nl {modid min max} {
	set w .ui$modid
	$w.f.numLevels configure -from $min -to $max
}

proc MeshView_set_minmax_numTet {modid min max} {
	set w .ui$modid
	$w.f.seedTet configure -from $min -to $max
}

proc MeshView_set_bounds {modid xmin xmax ymin ymax zmin zmax} {
    set w .ui$modid
    $w.f.clipX configure -from $xmin -to $xmax
    $w.f.clipNX configure -from $xmin -to $xmax
    $w.f.clipY configure -from $ymin -to $ymax
    $w.f.clipNY configure -from $ymin -to $ymax
    $w.f.clipZ configure -from $zmin -to $zmax
    $w.f.clipNZ configure -from $zmin -to $zmax
}


