
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
	global $this-elmMeas
	global $this-elmSwitch
	set $this-elmSwitch 0
	set $this-elmMeas 1
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
	pack $w.f -padx 4 -pady 4 -fill x
	set n "$this-c needexecute "

	scale $w.f.numLevels -variable $this-numLevels -from 0 -to 10 \
		-label "Number of Levels" -command $n \
		-orient horizontal
	pack $w.f.numLevels -side top -fill x

	scale $w.f.seedTet -variable $this-seedTet -from 0 -to 1 -command $n \
		-orient horizontal -label "Starting Tetrahedron"
	pack $w.f.seedTet -side top -fill both -anchor nw

	frame $w.f.levels
	pack $w.f.levels -side top
	label $w.f.levels.label -text "Show:"
	radiobutton $w.f.levels.on -text "All Levels" -relief flat \
		-variable $this-allLevels -value 0  -command $n
	radiobutton $w.f.levels.off -text "Outermost Level Only" -relief flat \
		-variable $this-allLevels -value 1  -command $n
	pack $w.f.levels.label $w.f.levels.on $w.f.levels.off -side left \
		-fill both -anchor nw


	frame $w.f.clips -relief groove -borderwidth 2 
	label $w.f.clips.label -text "Clipping surfaces"
	pack $w.f.clips -side top -expand 1 -fill x -pady 2
	
	range $w.f.clips.clipX -var_min $this-clipX -var_max $this-clipNX \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label X
	range $w.f.clips.clipY -var_min $this-clipY -var_max $this-clipNY \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Y
	range $w.f.clips.clipZ -var_min $this-clipZ -var_max $this-clipNZ \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Z
	pack $w.f.clips.clipX $w.f.clips.clipY \
		$w.f.clips.clipZ -side top -fill x


	frame $w.f.elms -relief groove -borderwidth 2
	pack $w.f.elms -side top -side left -padx 2 -pady 2
	label $w.f.elms.label -text "Quantifying Measures"
	pack $w.f.elms.label -side top

	frame $w.f.elms.switch
	pack $w.f.elms.switch -side top
	label $w.f.elms.switch.label -text "Show:"
	radiobutton $w.f.elms.switch.off -text "Off" -relief flat \
		-variable $this-elmSwitch -value 0  -command $n
	radiobutton $w.f.elms.switch.hil -text "Elements Only" \
		-relief flat \
		-variable $this-elmSwitch -value 1  -command $n
	radiobutton $w.f.elms.switch.only -text "Elements Hilited" \
		-relief flat  -variable $this-elmSwitch -value 2 \
		-command $n
        pack $w.f.elms.switch.label $w.f.elms.switch.off \
		$w.f.elms.switch.hil $w.f.elms.switch.only \
		-side left -fill both -anchor nw
	
	radiobutton $w.f.elms.volume -text "Volume" \
		-variable $this-elmMeas -value 1 -anchor w \
		-command $n
	radiobutton $w.f.elms.aspect -text "Aspect Ratio" \
		-variable $this-elmMeas -value 2 -anchor w \
		-command $n
	radiobutton $w.f.elms.size -text "Size v neighbors" \
		-variable $this-elmMeas -value 3 -anchor w \
		-command $n
	radiobutton $w.f.elms.err -text "Error" \
		-variable $this-elmMeas -value 4 -anchor w \
		-command $n
	pack $w.f.elms.volume $w.f.elms.aspect \
		$w.f.elms.size $w.f.elms.err -side top -expand 1 -fill x


	frame $w.f.elms.dummy
	pack $w.f.elms.dummy -side top -pady 2 -fill y

	range $w.f.elms.range -from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Measure \
		-var_min $this-mMin -var_max $this-mMax
	pack $w.f.elms.range -side left -fill x -expand 1

    }
    method set_minmax_nl {min max} {
	set w .ui$this
	global $w.f.numLevels
	$w.f.numLevels configure -from $min -to $max
    }
    method set_minmax_numTet {min max} {
	set w .ui$this
	global $w.f.seedTet
	$w.f.seedTet configure -from $min -to $max
    }
    method set_bounds {xmin xmax ymin ymax zmin zmax} {
	set w .ui$this
	global $w.f.clipX
	global $w.f.clipY
	global $w.f.clipZ
	global $w.f.clipNX
	global $w.f.clipNY
	global $w.f.clipNZ

	$w.f.clips.clipX configure -from $xmin -to $xmax
	$w.f.clips.clipY configure -from $ymin -to $ymax
	$w.f.clips.clipZ configure -from $zmin -to $zmax
    }

    method do_measure {min max} {
	global $this-mMin
	global $this-mMax
	set w .ui$this
	if {1 == 1} {
	    $w.f.elms.range configure -label "Volume"
	} elseif {1 == 2} {
	    $w.f.elms.range configure -label "Aspect Ratio"
	} elseif {1 == 3} {
	    $w.f.elms.range configure -label "Size v neighbor"
	} else {
	    $w.f.elms.range configure -label "Error"
	}
	$w.f.elms.range configure -from $min -to $max
	set $this-mMin $min
	set $this-mMax $max

    }

}


