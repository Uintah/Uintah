
catch {rename VolVis ""}

itcl_class Kurt_Vis_VolVis {
    inherit Module
    constructor {config} {
	set name VolVis
	set_defaults
    }
    method set_defaults {} {
	global $this-draw_mode
	set $this-draw_mode 0
	global $this-num_slices
	set $this-num_slices 32
	global $this-slice_transp
	set $this-slice_transp 0.075
	global $this-avail_tex
	set $this-avail_tex 2
	global $this-max_brick_dim
	set $this-max_brick_dim 16
	global $this-debug
	set $this-debug 0
	global $this-level
	set $this-level 0
	global $this-influence
	set $this-influence 0
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 250 300
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	global $this-draw_mode
	radiobutton $w.f.modep -text "CutPlane" -relief flat \
		-variable $this-draw_mode -value 0 \
		-command "$this DoCheck Mode $this-draw_mode"

	radiobutton $w.f.modev -text "VolRend" -relief flat \
		-variable $this-draw_mode -value 1 \
		-command "$this DoCheck Mode $this-draw_mode"

	radiobutton $w.f.modes -text "VolMIP" -relief flat \
		-variable $this-draw_mode -value 2 \
		-command "$this DoCheck Mode $this-draw_mode"

	radiobutton $w.f.moden -text "None" -relief flat \
		-variable $this-draw_mode -value 3 \
		-command "$this DoCheck Mode $this-draw_mode"
	set $this-draw_mode 1

	global $this-num_slices
	scale $w.f.nslice -variable $this-num_slices \
		-from 4 -to 1024 -label "Number of Slices" \
		-showvalue true \
		-orient horizontal \
		-command "$this DoScale NumSlices"
	pack $w.f.modep $w.f.modev $w.f.modes $w.f.moden $w.f.nslice \
		-side top -fill x

	global $this-max_brick_dim
	frame $w.f.dimframe -relief groove -border 2
	label $w.f.dimframe.l -text "Max brick dimension"
	pack $w.f.dimframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.dimframe.l -side top -fill x

	global $this-level
	frame $w.f.levelframe -relief groove -border 2
	label $w.f.levelframe.l -text "Max render level"
	pack $w.f.levelframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.levelframe.l -side top -fill x

	global $this-slice_transp
	scale $w.f.stransp -variable $this-slice_transp \
		-from 0.0 -to 1.0 -label "Slice Transparency" \
		-showvalue true -resolution 0.000001\
		-orient horizontal \
		-command "$this DoScale SliceTransp"
	global $this-influence
	scale $w.f.influence -variable $this-influence \
	    -from 0.0 -to 1.0 -label "Resolution widget influence" \
	    -resolution 0.01 -orient horizontal \
	    -command "$this DoScale Influence"

	global $this-avail_tex
	scale $w.f.availtex -variable $this-avail_tex \
		-from 1 -to 64 -label "Available Texture Memory (Mb)" \
		-showvalue true -orient horizontal
	pack $w.f.stransp $w.f.influence $w.f.availtex -side top -fill x

	checkbutton $w.f.debug -text debug -relief raised \
	    -variable $this-debug -offvalue 0 -onvalue 1
	pack $w.f.debug -side top -fill x
	# clear button for dave...

	button $w.f.clear -text "Clear" -command "$this-c Clear"
	pack $w.f.clear -side top -fill x

	frame $w.f.x
	button $w.f.x.plus -text "+" -command "$this-c MoveWidget xplus"
	label $w.f.x.label -text " X "
	button $w.f.x.minus -text "-" -command "$this-c MoveWidget xminus"
	pack $w.f.x.plus $w.f.x.label $w.f.x.minus -side left -fill x -expand 1
	pack $w.f.x -side top -fill x -expand 1

	frame $w.f.y
	button $w.f.y.plus -text "+" -command "$this-c MoveWidget yplus"
	label $w.f.y.label -text " Y "
	button $w.f.y.minus -text "-" -command "$this-c MoveWidget yminus"
	pack $w.f.y.plus $w.f.y.label $w.f.y.minus -side left -fill x -expand 1
	pack $w.f.y -side top -fill x -expand 1

	frame $w.f.z
	button $w.f.z.plus -text "+" -command "$this-c MoveWidget zplus"
	label $w.f.z.label -text " Z "
	button $w.f.z.minus -text "-" -command "$this-c MoveWidget zminus"
	pack $w.f.z.plus $w.f.z.label $w.f.z.minus -side left -fill x -expand 1
	pack $w.f.z -side top -fill x -expand 1

	button $w.f.exec -text "Execute" -command $n
	pack $w.f.exec -side top -fill x
    }


    method SetDims { val } {
	global $this-max_brick_dim
	set w .ui[modname]

	if {![winfo exists $w]} {
	    return
	}
	if {[winfo exists $w.f.dimframe.f]} {
	    destroy $w.f.dimframe.f
	}

	frame $w.f.dimframe.f -relief flat
	pack $w.f.dimframe.f -side top -fill x
	set f $w.f.dimframe.f
	for {set i 4} {$i <= $val} { set i [expr $i * 2]} {
	    radiobutton $f.brickdim$i -text $i -relief flat \
		-variable $this-max_brick_dim -value $i
	    pack $f.brickdim$i -side left -padx 2 -fill x
	}
    }
    method SetLevels { val } {
	global $this-level
	set w .ui[modname]

	if {![winfo exists $w]} {
	    return
	}
	
	if {[winfo exists $w.f.levelframe.f]} {
	    destroy $w.f.levelframe.f
	}

	if { $val == 0 } {
	    return
	}
	frame $w.f.levelframe.f -relief flat
	pack $w.f.levelframe.f -side top -fill x
	set f $w.f.levelframe.f
	scale $f.s -from 0 -to $val -tickinterval 1 \
	    -variable $this-level -orient horizontal
	pack $f.s -fill x
	
    }



    method DoScale {var val} {
	$this-c Set $var $val
    }
    
    method SetDim { val } {
	$this-c Set Dim $val
    }
    
    method DoCheck {cmd var} {
	global $var
	$this-c Set $cmd [set $var]
	if {[set $var] != 0} {
	    puts "turning off planes and widgets"
	}
	
    }
}
