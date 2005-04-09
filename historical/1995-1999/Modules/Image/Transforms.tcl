#
#

itcl_class Transforms {
    inherit Module
    constructor {config} {
	set name Transforms
	set_defaults
    }
    method set_defaults {} {
	global $this-coef
	set $this-coef 5000
	global $this-perr
	set $this-perr 3
	global $this-spread
	set $this-spread 20
	global $this-mode
	set $this-mode Coeffecients
	global $this-inverse
	set $this-inverse 0
	global $this-trans
	set $this-trans Haar
	$this-c needexecute
    }
    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 400 100    
	frame $w.f -width 400
	pack $w.f -padx 2 -pady 2 -fill x
	set n "$this-c needexecute "

	frame $w.f.r1
	pack $w.f.r1 -anchor nw

	checkbutton $w.f.v1 -text "Inverse Transform (other parameters become irrelavant)" -relief flat \
		-variable $this-inverse
	pack $w.f.v1 -side left
	
	entry $w.f.r1.n1 -relief sunken -width 17 -textvariable $this-coef
	entry $w.f.r1.n2 -relief sunken -width 15 -textvariable $this-perr
	entry $w.f.r1.n3 -relief sunken -width 15 -textvariable $this-spread

	frame $w.f.r1.v
	pack $w.f.r1.v -side top -fill x
	label $w.f.r1.v.lab -text "        Transform Parameters :       "
	pack $w.f.r1.v.lab -side left

	frame $w.f.r1.lab
	pack $w.f.r1.lab -side top -fill x
	label $w.f.r1.lab.hv -text "# of Non-zero Coef's"
	label $w.f.r1.lab.mv -text "Max Pixel Error"
	label $w.f.r1.lab.lv -text "   Spread"
	pack $w.f.r1.lab.hv $w.f.r1.lab.mv $w.f.r1.lab.lv -side left

	pack $w.f.r1.n1 $w.f.r1.n2 $w.f.r1.n3 -side left

	make_labeled_radio $w.trans "Transform:" "" left $this-trans \
		{Haar Hadamard Walsh}
	pack $w.trans -side top
	entry $w.f.ted -textvariable $this-trans -width 40 \
		-borderwidth 2 -relief sunken

	make_labeled_radio $w.mode "Compression Search mode:" "" left $this-mode \
		{Coeffecients PixelError}
	pack $w.mode -side top
	entry $w.f.ted2 -textvariable $this-mode -width 40 \
		-borderwidth 2 -relief sunken


    }
}

