
catch {rename VolVis ""}

itcl_class Kurt_VolVis_VolVis {
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
	set $this-max_brick_dim 32
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
	radiobutton $w.f.dimframe.brickdim4 -text 4 -relief flat \
	    -variable $this-max_brick_dim -value 4 \
	    -command "$this SetDim $this-max_brick_dim"
	radiobutton $w.f.dimframe.brickdim8 -text 8 -relief flat \
	    -variable $this-max_brick_dim -value 8 \
	    -command "$this SetDim $this-max_brick_dim"
	radiobutton $w.f.dimframe.brickdim16 -text 16 -relief flat \
	    -variable $this-max_brick_dim -value 16 \
	    -command "$this SetDim $this-max_brick_dim"
	radiobutton $w.f.dimframe.brickdim32 -text 32 -relief flat \
	    -variable $this-max_brick_dim -value 32 \
	    -command "$this SetDim $this-max_brick_dim"
	radiobutton $w.f.dimframe.brickdim64 -text 64 -relief flat \
	    -variable $this-max_brick_dim -value 64 \
	    -command "$this SetDim $this-max_brick_dim"
	radiobutton $w.f.dimframe.brickdim128 -text 128 -relief flat \
	    -variable $this-max_brick_dim -value 128 \
	    -command "$this SetDim $this-max_brick_dim"

	pack $w.f.dimframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.dimframe.l -side top
	pack $w.f.dimframe.brickdim4 $w.f.dimframe.brickdim8 \
	    $w.f.dimframe.brickdim16 $w.f.dimframe.brickdim32 \
	    $w.f.dimframe.brickdim64 $w.f.dimframe.brickdim128 \
	    -side left -padx 2 -fill x

	global $this-slice_transp
	scale $w.f.stransp -variable $this-slice_transp \
		-from 0.0 -to 1.0 -label "Slice Transparency" \
		-showvalue true -resolution 0.000001\
		-orient horizontal \
		-command "$this DoScale SliceTransp"

	global $this-avail_tex
	scale $w.f.availtex -variable $this-avail_tex \
		-from 1 -to 64 -label "Available Texture Memory (Mb)" \
		-showvalue true -orient horizontal
	pack $w.f.stransp $w.f.availtex -side top -fill x

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
