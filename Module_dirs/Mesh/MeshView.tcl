
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
	set $this-elmSwitch 1
	set $this-elmMeas 1
	global $this-tech
	set $this-tech 0
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

	frame $w.f.man -relief groove -borderwidth 3
	pack $w.f.man -side top -padx 2 -pady 2
	label $w.f.man.label -text "Manipulative Techniques"
	pack $w.f.man.label -side top

	scale $w.f.man.numLevels -variable $this-numLevels -from 0 -to 10 \
		-label "Number of Levels" -command $n \
		-orient horizontal
	pack $w.f.man.numLevels -side top -fill x

	scale $w.f.man.seedTet -variable $this-seedTet -from 0 -to 1 \
		-command $n \
		-orient horizontal -label "Starting Tetrahedron"
	pack $w.f.man.seedTet -side top -fill both -anchor nw

	frame $w.f.man.levels
	pack $w.f.man.levels -side top
	label $w.f.man.levels.label -text "Show:"
	radiobutton $w.f.man.levels.on -text "All Levels" -relief flat \
		-variable $this-allLevels -value 0  -command $n
	radiobutton $w.f.man.levels.off -text "Outermost Level Only" \
		-relief flat \
		-variable $this-allLevels -value 1  -command $n
	pack $w.f.man.levels.label $w.f.man.levels.on $w.f.man.levels.off \
		-side left -fill both -anchor nw


	frame $w.f.man.clips -relief groove -borderwidth 2 
	label $w.f.man.clips.label -text "Clipping surfaces"
	pack $w.f.man.clips -side top -expand 1 -fill x -pady 2
	pack $w.f.man.clips.label -side top

	range $w.f.man.clips.clipX -var_min $this-clipX -var_max $this-clipNX \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label X
	range $w.f.man.clips.clipY -var_min $this-clipY -var_max $this-clipNY \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Y
	range $w.f.man.clips.clipZ -var_min $this-clipZ -var_max $this-clipNZ \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Z
	pack $w.f.man.clips.clipX $w.f.man.clips.clipY \
		$w.f.man.clips.clipZ -side top -fill x


	frame $w.f.which
	pack $w.f.which -side top
	label $w.f.which.label -text "Technique: "
	pack $w.f.which.label -side top

	radiobutton $w.f.which.man -text "Manipulative" -relief flat \
		-variable $this-tech -value 0 -command $n \
		-anchor w
	radiobutton $w.f.which.elm -text "Quantifying" -relief flat \
		-variable $this-tech -value 1 -command $n \
		-anchor w
 	pack $w.f.which.man $w.f.which.elm -side top -expand 1 -fill x

	frame $w.f.elms -relief groove -borderwidth 3
	pack $w.f.elms -side left -padx 2 -pady 2
	label $w.f.elms.label -text "Quantifying Measures"
	pack $w.f.elms.label -side top

	frame $w.f.elms.switch
	pack $w.f.elms.switch -side top
	label $w.f.elms.switch.label -text "Show:"
	radiobutton $w.f.elms.switch.hil -text "Elements Only" \
		-relief flat \
		-variable $this-elmSwitch -value 1  -command $n
	radiobutton $w.f.elms.switch.only -text "Elements Hilited" \
		-relief flat  -variable $this-elmSwitch -value 2 \
		-command $n
        pack $w.f.elms.switch.label \
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
		-orient horizontal -command $n -label Volume \
		-var_min $this-mMin -var_max $this-mMax -resolution 0.000001
	pack $w.f.elms.range -side left -fill x -expand 1

    }
    method set_minmax_nl {min max} {
	set w .ui$this
	global $w.f.numLevels
	$w.f.man.numLevels configure -from $min -to $max
    }
    method set_minmax_numTet {min max} {
	set w .ui$this
	global $w.f.seedTet
	$w.f.man.seedTet configure -from $min -to $max
    }
    method set_bounds {xmin xmax ymin ymax zmin zmax} {
	set w .ui$this
	global $w.f.clipX
	global $w.f.clipY
	global $w.f.clipZ
	global $w.f.clipNX
	global $w.f.clipNY
	global $w.f.clipNZ

	$w.f.man.clips.clipX configure -from $xmin -to $xmax
	$w.f.man.clips.clipY configure -from $ymin -to $ymax
	$w.f.man.clips.clipZ configure -from $zmin -to $zmax
    }

    method do_measure {min max} {
	global $this-mMin
	global $this-mMax
	global $this-elmMeas
	set w .ui$this

	set a [set $this-elmMeas]

puts -nonewline "elmMeas = "
puts $a
	if {$a == 1} {
	    $w.f.elms.range configure -label "Volume"
	} elseif {$a == 2} {
	    $w.f.elms.range configure -label "Aspect Ratio"
	} elseif {$a == 3} {
	    $w.f.elms.range configure -label "Size v neighbor"
	} else {
	    $w.f.elms.range configure -label "Error"
	}
	$w.f.elms.range configure -from $min -to $max
	set $this-mMin $min
	set $this-mMax $max

    }

}


