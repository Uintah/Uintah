
proc uiStreamline {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    frame $w.f 
    pack $w.f -padx 2 -pady 2
    set n "$modid needexecute "

    frame $w.f.widgets
    pack $w.f.widgets -side top -fill x
    label $w.f.widgets.label -text "Source:"
    pack $w.f.widgets.label -padx 2 -side left
    global source
    set source($modid) "Line"
    radiobutton $w.f.widgets.point -text Point -relief flat \
	-variable source($modid) -value Point -command $n
    radiobutton $w.f.widgets.line -text Line -relief flat \
	-variable source($modid) -value Line -command $n
    radiobutton $w.f.widgets.square -text Square -relief flat \
	-variable source($modid) -value Square -command $n
    pack $w.f.widgets.point $w.f.widgets.line $w.f.widgets.square -side left

    frame $w.f.marker
    pack $w.f.marker -side top -fill x
    label $w.f.marker.label -text "Marker:"
    pack $w.f.marker.label -padx 2 -side left
    global marker
    set marker($modid) "Line"
    radiobutton $w.f.marker.line -text Line -relief flat \
	-variable marker($modid) -value Line -command $n
    radiobutton $w.f.marker.ribbon -text Ribbon -relief flat \
	-variable marker($modid) -value Ribbon -command $n
    pack $w.f.marker.line $w.f.marker.ribbon -side left

    global radius
    set radius($modid) 0
    fscale $w.f.lineradius -variable radius($modid) -digits 3 \
	-from 0.0 -to .10 -label "Line Radius Scale:" \
	-resolution .01 -showvalue true -tickinterval .02 \
	-activeforeground SteelBlue2 -orient horizontal
    pack $w.f.lineradius -fill x -pady 2

    frame $w.f.alg
    pack $w.f.alg -side top -fill x
    label $w.f.alg.label -text "Algorithm:"
    pack $w.f.alg.label -padx 2 -side left -anchor nw
    global algorithm
    set algorithm($modid) "Euler"
    radiobutton $w.f.alg.euler -text "Euler" -relief flat \
	-variable algorithm($modid) -value "Euler" \
	-command $n
    radiobutton $w.f.alg.rk4 -text "4th Order Runge Kutta" -relief flat \
	-variable algorithm($modid) -value "RK4" \
	-command $n
    radiobutton $w.f.alg.sf -text "Stream Function" -relief flat \
	-variable algorithm($modid) -value "Stream Function" \
	-command $n
    pack $w.f.alg.euler $w.f.alg.rk4 $w.f.alg.sf -side top -anchor w

    global stepsize
    set stepsize($modid) 0.1
    fscale $w.f.stepsize -variable stepsize($modid) -digits 3 \
	-from 0.0 -to 1.0 -label "Step size:" \
	-resolution .01 -showvalue true -tickinterval .2 \
	-activeforeground SteelBlue2 -orient horizontal
    pack $w.f.stepsize -fill x -pady 2

    global maxsteps
    set maxsteps($modid) 100
    fscale $w.f.maxsteps -variable maxsteps($modid) \
	-from 0 -to 1000 -label "Maximum steps:" \
	-showvalue true -tickinterval 200 \
	-activeforeground SteelBlue2 -orient horizontal
    pack $w.f.maxsteps -fill x -pady 2
}
