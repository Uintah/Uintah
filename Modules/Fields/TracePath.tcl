itcl_class TracePath {
    inherit Module
    constructor {config} {
	set name TracePath
	set_defaults
    }
    method set_defaults {} {
	global $this-swapXZ
	global $this-thresh
	global $this-partial
	global $this-updatelines
	global $this-sx $this-sy $this-sz
	global $this-eax $this-eay $this-eaz
	global $this-ebx $this-eby $this-ebz
	global $this-ecx $this-ecy $this-ecz
	global $this-tclAlpha $this-tclBeta
	set $this-swapXZ 0
	set $this-thresh 2.3
	set $this-partial 0
	set $this-updatelines 0
	set $this-sx 0
	set $this-sy 0
	set $this-sz 0
	set $this-eax 0
	set $this-eay 0
	set $this-eaz 0
	set $this-ebx 0
	set $this-eby 0
	set $this-ebz 0
	set $this-ecx 0
	set $this-ecy 0
	set $this-ecz 0
	set $this-tclAlpha 0.99
	set $this-tclBeta 0.99
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	global $this-swapXZ
	global $this-partial
	global $this-updatelines
	global $this-sx $this-sy $this-sz
	global $this-eax $this-eay $this-eaz
	global $this-ebx $this-eby $this-ebz
	global $this-ecx $this-ecy $this-ecz
	global $this-tclAlpha
	global $this-tclBeta
	checkbutton $w.xz -text "Swap X/Z" -variable $this-swapXZ
	checkbutton $w.par -text "Output Partial Vol" -variable $this-partial
	checkbutton $w.ul -text "Update Lines" -variable $this-updatelines
	button $w.gp -text "Generate Paths" -command "$this-c gen_paths"
	button $w.cp -text "Clear Paths" -command "$this-c clear_paths"
	button $w.ae -text "Add Endpoint" -command "$this-c add_endpt"
	button $w.ce -text "Clear Endpoints" -command "$this-c clear_endpts"
	button $w.al -text "Add Lines" -command "$this-c add_lines"
	button $w.cl -text "Clear Lines" -command "$this-c clear_lines"
	button $w.nf -text "New Field" -command "$this-c new_field"
	button $w.abort -text "Abort" -command "$this-c abort"
	scale $w.thresh -label "Threshold" -variable "$this-thresh" \
		-orient horizontal -from 1.0 -to 5.0 -showvalue true \
		-resolution 0.1
	scale $w.alpha -label "Alpha" -variable "$this-tclAlpha" \
		-orient horizontal -from 0.00 -to 1.00 -showvalue true \
		-resolution 0.01
	scale $w.beta -label "Beta" -variable "$this-tclBeta" \
		-orient horizontal -from 0.00 -to 1.00 -showvalue true \
		-resolution 0.01
	pack $w.xz $w.par $w.ul $w.gp $w.cp $w.ae $w.ce $w.al $w.cl $w.nf \
		$w.abort $w.thresh $w.alpha $w.beta -side top -fill both \
		-expand 1
	frame $w.start -relief raised -bd 2
	frame $w.start.h
	button $w.start.h.s -text "Get" -command "$this-c getstart"
	label $w.start.h.l -text "Start position"
	button $w.start.h.g -text "Set" -command "$this-c setstart"
	pack $w.start.h.s $w.start.h.l $w.start.h.g -side left -pady 2 -padx 4
	frame $w.start.x
	label $w.start.x.l -text "X: "
	entry $w.start.x.v -width 7 -relief sunken -bd 2 -textvariable $this-sx
	pack $w.start.x.l $w.start.x.v -side left
	frame $w.start.y
	label $w.start.y.l -text "Y: "
	entry $w.start.y.v -width 7 -relief sunken -bd 2 -textvariable $this-sy
	pack $w.start.y.l $w.start.y.v -side left
	frame $w.start.z
	label $w.start.z.l -text "Z: "
	entry $w.start.z.v -width 7 -relief sunken -bd 2 -textvariable $this-sz
	pack $w.start.z.l $w.start.z.v -side left
	pack $w.start.h $w.start.x $w.start.y $w.start.z -side top

	frame $w.endpta -relief raised -bd 2
	frame $w.endpta.h
	button $w.endpta.h.s -text "Get" -command "$this-c getendpta"
	label $w.endpta.h.l -text "EndPt 1 position"
	button $w.endpta.h.g -text "Set" -command "$this-c setendpta"
	pack $w.endpta.h.s $w.endpta.h.l $w.endpta.h.g -side left -pady 2 -padx 4
	frame $w.endpta.x
	label $w.endpta.x.l -text "X: "
	entry $w.endpta.x.v -width 7 -relief sunken -bd 2 -textvariable $this-eax
	pack $w.endpta.x.l $w.endpta.x.v -side left
	frame $w.endpta.y
	label $w.endpta.y.l -text "Y: "
	entry $w.endpta.y.v -width 7 -relief sunken -bd 2 -textvariable $this-eay
	pack $w.endpta.y.l $w.endpta.y.v -side left
	frame $w.endpta.z
	label $w.endpta.z.l -text "Z: "
	entry $w.endpta.z.v -width 7 -relief sunken -bd 2 -textvariable $this-eaz
	pack $w.endpta.z.l $w.endpta.z.v -side left
	pack $w.endpta.h $w.endpta.x $w.endpta.y $w.endpta.z -side top

	frame $w.endptb -relief raised -bd 2
	frame $w.endptb.h
	button $w.endptb.h.s -text "Get" -command "$this-c getendptb"
	label $w.endptb.h.l -text "EndPt 2 position"
	button $w.endptb.h.g -text "Set" -command "$this-c setendptb"
	pack $w.endptb.h.s $w.endptb.h.l $w.endptb.h.g -side left -pady 2 -padx 4
	frame $w.endptb.x
	label $w.endptb.x.l -text "X: "
	entry $w.endptb.x.v -width 7 -relief sunken -bd 2 -textvariable $this-ebx
	pack $w.endptb.x.l $w.endptb.x.v -side left
	frame $w.endptb.y
	label $w.endptb.y.l -text "Y: "
	entry $w.endptb.y.v -width 7 -relief sunken -bd 2 -textvariable $this-eby
	pack $w.endptb.y.l $w.endptb.y.v -side left
	frame $w.endptb.z
	label $w.endptb.z.l -text "Z: "
	entry $w.endptb.z.v -width 7 -relief sunken -bd 2 -textvariable $this-ebz
	pack $w.endptb.z.l $w.endptb.z.v -side left
	pack $w.endptb.h $w.endptb.x $w.endptb.y $w.endptb.z -side top

	frame $w.endptc -relief raised -bd 2
	frame $w.endptc.h
	button $w.endptc.h.s -text "Get" -command "$this-c getendptc"
	label $w.endptc.h.l -text "EndPt 3 position"
	button $w.endptc.h.g -text "Set" -command "$this-c setendptc"
	pack $w.endptc.h.s $w.endptc.h.l $w.endptc.h.g -side left -pady 2 -padx 4
	frame $w.endptc.x
	label $w.endptc.x.l -text "X: "
	entry $w.endptc.x.v -width 7 -relief sunken -bd 2 -textvariable $this-ecx
	pack $w.endptc.x.l $w.endptc.x.v -side left
	frame $w.endptc.y
	label $w.endptc.y.l -text "Y: "
	entry $w.endptc.y.v -width 7 -relief sunken -bd 2 -textvariable $this-ecy
	pack $w.endptc.y.l $w.endptc.y.v -side left
	frame $w.endptc.z
	label $w.endptc.z.l -text "Z: "
	entry $w.endptc.z.v -width 7 -relief sunken -bd 2 -textvariable $this-ecz
	pack $w.endptc.z.l $w.endptc.z.v -side left
	pack $w.endptc.h $w.endptc.x $w.endptc.y $w.endptc.z -side top
	pack $w.start $w.endpta $w.endptb $w.endptc -side top -fill x
    }
}

