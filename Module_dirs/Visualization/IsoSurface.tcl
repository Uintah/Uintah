
proc uiIsoSurface {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    frame $w.f 
    pack $w.f -padx 2 -pady 2
    set n "$modid needexecute "

    global have_seedpoint,$modid
    set have_seedpoint,$modid 1
    frame $w.f.seedpoint
    pack $w.f.seedpoint
    label $w.f.seedpoint.label -text "Algorithm: "
    radiobutton $w.f.seedpoint.value -text Value -relief flat \
	    -variable have_seedpoint,$modid -value 0 \
	    -command "$w.f.seedpoint.w3d configure -state disabled ; $n"
    radiobutton $w.f.seedpoint.seedpoint -text Seedpoint -relief flat \
	    -variable have_seedpoint,$modid -value 1 \
	    -command "$w.f.seedpoint.w3d configure -state normal ; $n"

    global do_3dwidget,$modid
    set do_3dwidget,$modid 1
    checkbutton $w.f.seedpoint.w3d -text "3D widget" -relief flat \
	    -variable do_3dwidget,$modid -command $n
    pack $w.f.seedpoint.label $w.f.seedpoint.value \
	    $w.f.seedpoint.seedpoint $w.f.seedpoint.w3d -side left

    global isoval,$modid
    fscale $w.f.isoval -variable isoval,$modid -digits 3 \
	    -from 0.0 -to 1.0 -label "IsoValue:" \
	    -resolution .01 -showvalue true -tickinterval .2 \
	    -activeforeground SteelBlue2 -orient horizontal
    pack $w.f.isoval -side top -fill x

    makePoint $w.f.seed "Seed Point" seedpoint,$modid $n
    pack $w.f.seed -fill x

    button $w.f.emit_surface -text "Emit Surface" -command "emitIsoSurface $modid"
    pack $w.f.emit_surface
}

proc emitIsoSurface {modid} {
    global emit_surface,$modid
    set emit_surface,$modid 1
    $modid needexecute
}

proc IsoSurface_set_minmax {modid min max} {
    set w .ui$modid	
    $w.f.isoval configure -from $min -to $max
}
