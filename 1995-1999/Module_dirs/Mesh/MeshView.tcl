
catch {rename MeshView ""}

itcl_class MeshView {
    inherit Module
    public numNod 0
    public numTet 0
    public seed 0
    public vol 0
    public asp 0
    public edit 0
    public editVol 0
    public editAsp 0

    constructor {config} {
	set name MeshView
	set_defaults
    }
    method set_defaults {} {
	global $this-allLevels $this-elmMeas $this-elmSwitch $this-tech \
		$this-inside $this-render $this-editMode $this-numNod \
		$this-numTet $this-display $this-radius $this-select
	set $this-allLevels 1
	set $this-elmSwitch 1
	set $this-elmMeas 1
	set $this-tech 0
	set $this-inside 1
	set $this-render 0
	set $this-editMode 0
	set $this-display 0
	set $this-radius 0.025
	set $this-select 0
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
	pack $w.f  -fill x
	set n "$this-c needexecute "

	frame $w.f.left
	pack $w.f.left -side left -anchor w -expand 1 -fill both

	frame $w.f.left.which
	pack $w.f.left.which -side top -anchor w
	label $w.f.left.which.label -text "Technique: "
	pack $w.f.left.which.label -side top 

	radiobutton $w.f.left.which.man -text "Manipulative" -relief flat \
		-variable $this-tech -value 0 -command $n \
		-anchor w
	radiobutton $w.f.left.which.elm -text "Quantifying" -relief flat \
		-variable $this-tech -value 1 -command $n \
		-anchor w
	radiobutton $w.f.left.which.edt -text "Editing" -relief flat \
		-variable $this-tech -value 2 -command $n \
		-anchor w
 	pack $w.f.left.which.man $w.f.left.which.elm $w.f.left.which.edt \
		-side left -expand 1 -fill x -anchor nw

	frame $w.f.left.render -relief groove -borderwidth 3
	pack $w.f.left.render -side top -anchor w -expand 1 -fill both
	frame $w.f.left.render.switch
	pack $w.f.left.render.switch -side top -anchor w
	label $w.f.left.render.switch.label -text "Display:"
	radiobutton $w.f.left.render.switch.on -text "Production" \
		-relief flat -variable $this-display -value 1 -command $n
	radiobutton $w.f.left.render.switch.off -text "Normal" -relief flat \
		-variable $this-display -value 0 -command $n
	pack $w.f.left.render.switch.label $w.f.left.render.switch.on \
		$w.f.left.render.switch.off \
		-side left -fill both -anchor nw \
		-expand 1
	scale $w.f.left.render.slider -variable $this-radius -from 0 \
		-to 2 -orient horizontal -command $n \
		-showvalue true -resolution 0.001 -label "Cylinder width"
	pack $w.f.left.render.slider -side left -fill both -anchor nw \
		-expand 1

	frame $w.f.left.man -relief groove -borderwidth 3
	pack $w.f.left.man -side left -expand 1 -fill both -anchor w
	label $w.f.left.man.label -text "Manipulative Techniques"
	pack $w.f.left.man.label -side top -expand 1 -fill x

	scale $w.f.left.man.numLevels -variable $this-numLevels -from 0 \
		-to 10 -label "Number of Levels" -command $n \
		-orient horizontal
	pack $w.f.left.man.numLevels -side top -fill x -expand 1

	frame $w.f.left.man.levels 
	pack $w.f.left.man.levels -side top -anchor e
	label $w.f.left.man.levels.label -text "Show:"
	radiobutton $w.f.left.man.levels.on -text "All Levels" -relief flat \
		-variable $this-allLevels -value 0  -command $n
	radiobutton $w.f.left.man.levels.off -text "Outermost Level Only" \
		-relief flat \
		-variable $this-allLevels -value 1  -command $n
	pack $w.f.left.man.levels.label $w.f.left.man.levels.on \
		$w.f.left.man.levels.off \
		-side left -fill both -anchor nw -expand 1
	
	frame $w.f.left.man.surface
	pack $w.f.left.man.surface -side top
	label $w.f.left.man.surface.label -text "Render:"
	radiobutton $w.f.left.man.surface.ext -text "Surface Only" \
		-relief flat \
		-variable $this-render -value 0 -command $n
	radiobutton $w.f.left.man.surface.int -text "Entire Volume" \
		-relief flat \
		-variable $this-render -value 1 -command $n
	pack $w.f.left.man.surface.label $w.f.left.man.surface.ext \
		$w.f.left.man.surface.int -side left -fill both -anchor nw \
		-expand 1

	frame $w.f.left.man.clips -relief groove -borderwidth 2 
	label $w.f.left.man.clips.label -text "Clipping surfaces"
	pack $w.f.left.man.clips -side top -expand 1 -fill x -pady 2
	pack $w.f.left.man.clips.label -side top

	range $w.f.left.man.clips.clipX -var_min $this-clipX \
		-var_max $this-clipNX \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label X -resolution 0.01
	range $w.f.left.man.clips.clipY -var_min $this-clipY \
		-var_max $this-clipNY \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Y -resolution 0.01
	range $w.f.left.man.clips.clipZ -var_min $this-clipZ \
		-var_max $this-clipNZ \
		-from -1 -to 1 -showvalue true \
		-orient horizontal -command $n -label Z -resolution 0.01
	pack $w.f.left.man.clips.clipX $w.f.left.man.clips.clipY \
		$w.f.left.man.clips.clipZ -side top -fill x -expand 1


	frame $w.f.right
	pack $w.f.right -side right -anchor e \
		-expand 1 -fill both

	frame $w.f.right.elms -relief groove -borderwidth 3
	pack $w.f.right.elms -side top -anchor ne \
		-expand 1 -fill both
	label $w.f.right.elms.label -text "Quantifying Measures"
	pack $w.f.right.elms.label -side top 

	frame $w.f.right.elms.switch
	pack $w.f.right.elms.switch -side top
	label $w.f.right.elms.switch.label -text "Show:"
	radiobutton $w.f.right.elms.switch.only -text "Elements Only" \
		-relief flat \
		-variable $this-elmSwitch -value 1  -command $n
	radiobutton $w.f.right.elms.switch.hil -text "Elements Hilited" \
		-relief flat  -variable $this-elmSwitch -value 2 \
		-command $n
        pack $w.f.right.elms.switch.label \
		$w.f.right.elms.switch.only $w.f.right.elms.switch.hil \
		-side left -fill both -anchor ne
	
	frame $w.f.right.elms.inside
	pack $w.f.right.elms.inside -side top
	label $w.f.right.elms.inside.label -text "Show elements"
	radiobutton $w.f.right.elms.inside.in -text "inside" \
		-relief flat \
		-variable $this-inside -value 1 -command $n
	radiobutton $w.f.right.elms.inside.out -text "outside" \
		-relief flat \
		-variable $this-inside -value 0 -command $n
	label $w.f.right.elms.inside.lab2 -text "of the range"
	pack $w.f.right.elms.inside.label $w.f.right.elms.inside.in \
		$w.f.right.elms.inside.out $w.f.right.elms.inside.lab2 \
		-side left -fill both -anchor nw

	radiobutton $w.f.right.elms.volume -text "Volume" \
		-variable $this-elmMeas -value 1 -anchor w \
		-command $n
	radiobutton $w.f.right.elms.aspect -text "Aspect Ratio" \
		-variable $this-elmMeas -value 2 -anchor w \
		-command $n
	radiobutton $w.f.right.elms.size -text "Size v neighbors" \
		-variable $this-elmMeas -value 3 -anchor w \
		-command $n
	radiobutton $w.f.right.elms.err -text "Error" \
		-variable $this-elmMeas -value 4 -anchor w \
		-command $n
	pack $w.f.right.elms.volume $w.f.right.elms.aspect \
		$w.f.right.elms.size $w.f.right.elms.err -side top \
		-expand 1 -fill x

	frame $w.f.right.edit -relief groove -borderwidth 3
	pack $w.f.right.edit -side top -expand 1 -fill both
	label $w.f.right.edit.label -text "Editing"
	pack $w.f.right.edit.label -side top

	frame $w.f.right.edit.opt 
	radiobutton $w.f.right.edit.opt.add -text "Add node" \
		-relief flat -variable $this-editMode -value 1 \
		-command $n -anchor w
	radiobutton $w.f.right.edit.opt.del -text "Delete element and nodes" \
		-relief flat -variable $this-editMode -value 2 \
		-command $n -anchor w
	radiobutton $w.f.right.edit.opt.node -text "Delete nearest node" \
		-relief flat -variable $this-editMode -value 3 \
		-command $n -anchor w
	pack $w.f.right.edit.opt $w.f.right.edit.opt.add \
		$w.f.right.edit.opt.del \
		$w.f.right.edit.opt.node \
		-side top -fill both -anchor w

	frame $w.f.right.select -relief groove -borderwidth 3
	pack $w.f.right.select -side top -anchor w -expand 1 -fill both
	label $w.f.right.select.label -text "Selecting: "
	radiobutton $w.f.right.select.seed -text "Seed" -relief flat \
		-variable $this-select -value 0 -command $n
	radiobutton $w.f.right.select.edit -text "Edit" -relief flat \
		-variable $this-select -value 1 -command $n
	pack $w.f.right.select.label $w.f.right.select.seed \
		$w.f.right.select.edit -side left -anchor w -fill both \
		-expand 1


	frame $w.f.right.info -relief groove -borderwidth 3
	pack $w.f.right.info -side bottom -expand 1 -fill both 
	label $w.f.right.info.label -text "Information"
	pack $w.f.right.info.label -side top

	frame $w.f.right.info.numNodes
	pack $w.f.right.info.numNodes -side top -anchor w
	label $w.f.right.info.numNodes.lab -text "Number of nodes: " -anchor w
	label $w.f.right.info.numNodes.num -text [format %d $numNod]
	pack $w.f.right.info.numNodes.lab $w.f.right.info.numNodes.num -side left

	frame $w.f.right.info.numTet
	pack $w.f.right.info.numTet -side top -anchor w
	label $w.f.right.info.numTet.lab -text "Number of elements: " -anchor w
	label $w.f.right.info.numTet.num -text [format %d  $numTet]
	pack $w.f.right.info.numTet.lab $w.f.right.info.numTet.num -side left

	frame $w.f.right.info.elem
	pack $w.f.right.info.elem -side top -anchor w -expand 1 -fill x

	frame $w.f.right.info.elem.seed 
	pack $w.f.right.info.elem.seed -side left -anchor w -expand 1 \
		-fill both
	label $w.f.right.info.elem.seed.lab \
		-text "               Seed element " \
		-anchor w
	pack $w.f.right.info.elem.seed.lab -side top -anchor w

	frame $w.f.right.info.elem.seed.elem
	pack $w.f.right.info.elem.seed.elem -side top -anchor w
	label $w.f.right.info.elem.seed.elem.lab -text "Element:       " \
		-anchor w
	label $w.f.right.info.elem.seed.elem.num -text [format %d $seed]
	pack $w.f.right.info.elem.seed.elem.lab \
		$w.f.right.info.elem.seed.elem.num -side left -anchor w

	frame $w.f.right.info.elem.seed.vol 
	pack $w.f.right.info.elem.seed.vol -side top -anchor w
	label $w.f.right.info.elem.seed.vol.lab -text "Volume:        " \
		-anchor w
	label $w.f.right.info.elem.seed.vol.num -text [format %d $vol]
	pack $w.f.right.info.elem.seed.vol.lab \
		$w.f.right.info.elem.seed.vol.num -side left -anchor w

	frame $w.f.right.info.elem.seed.asp
	pack $w.f.right.info.elem.seed.asp -side top -anchor w
	label $w.f.right.info.elem.seed.asp.lab -text "Aspect ratio: " \
		-anchor w
	label $w.f.right.info.elem.seed.asp.num -text [format %d $asp]
	pack $w.f.right.info.elem.seed.asp.lab \
		$w.f.right.info.elem.seed.asp.num -side left -anchor w

	frame $w.f.right.info.elem.edit
	pack $w.f.right.info.elem.edit -side right -anchor e -expand 1\
		-fill both
	label $w.f.right.info.elem.edit.lab -text "Edit element" -anchor w
	pack $w.f.right.info.elem.edit.lab -side top -anchor w

	frame $w.f.right.info.elem.edit.elem
	pack $w.f.right.info.elem.edit.elem -side top -anchor w
	label $w.f.right.info.elem.edit.elem.num -text [format %d $edit]
	pack $w.f.right.info.elem.edit.elem.num -side left -anchor w

	frame $w.f.right.info.elem.edit.vol
	pack $w.f.right.info.elem.edit.vol -side top -anchor w
	label $w.f.right.info.elem.edit.vol.num -text [format %d $editVol]
	pack $w.f.right.info.elem.edit.vol.num -side left -anchor w

	frame $w.f.right.info.elem.edit.asp
	pack $w.f.right.info.elem.edit.asp -side top -anchor w
	label $w.f.right.info.elem.edit.asp.num -text [format %d $editAsp]
	pack $w.f.right.info.elem.edit.asp.num -side left -anchor w
    }
    method set_minmax_nl {min max} {
	set w .ui$this
	global $w.f.numLevels
	$w.f.left.man.numLevels configure -from $min -to $max
    }
    method set_bounds {xmin xmax ymin ymax zmin zmax} {
	set w .ui$this
	global $this-clipX
	global $this-clipY
	global $this-clipZ
	global $this-clipNX
	global $this-clipNY
	global $this-clipNZ

	set a [expr $xmax - $xmin]
	set b [expr $a / 5]
	$w.f.left.man.clips.clipX configure -from $xmin -to $xmax \
		-tickinterval $b

	set a [expr $ymax - $ymin]
	set b [expr $a / 5]	
	$w.f.left.man.clips.clipY configure -from $ymin -to $ymax \
		-tickinterval $b
	set a [expr $zmax - $zmin]
	set b [expr $a / 5]	
	$w.f.left.man.clips.clipZ configure -from $zmin -to $zmax \
		-tickinterval $b
    }
    method set_info {nodes tetras start volVal aspVal toedit vol2 asp2} {
	set numNod $nodes
	set numTet $tetras
	set seed $start
	set vol $volVal
	set asp $aspVal
	set edit $toedit
	set editVol $vol2
	set editAsp $asp2

	set w .ui$this

	$w.f.right.info.numTet.num configure -text [format %d $numTet]
	$w.f.right.info.numNodes.num configure -text [format %d $numNod]
	$w.f.right.info.elem.seed.elem.num configure -text [format %d $seed]
	$w.f.right.info.elem.seed.vol.num configure -text [format %.3f $vol]
	$w.f.right.info.elem.seed.asp.num configure -text [format %.3f $asp]
	$w.f.right.info.elem.edit.elem.num configure -text [format %d $edit]
	$w.f.right.info.elem.edit.vol.num configure \
		-text [format %.3f $editVol]
	$w.f.right.info.elem.edit.asp.num configure \
		-text [format %.3f $editAsp]
    }
}


