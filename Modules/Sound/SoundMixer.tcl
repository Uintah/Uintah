
proc ModulePrologue {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
	raise $w
	return true;
    }
    toplevel $w
    wm minsize $w 20 20
    return false;
}

proc uiSoundMixer {modid} {
    if {[ModulePrologue $modid]} {
	return;
    }
    set w .ui$modid
    frame $w.f
    pack $w.f -padx 2 -pady 2 -fill both
    #set n $modid needexecute
    global overall_gain,$modid
    set overall_gain,$modid 1.0
    fscale $w.f.overall_gain -variable overall_gain,$modid -digits 2 \
	    -from 0.0 -to 2.0 -label "Overall Gain:" \
	    -resolution 0 -showvalue true \
	    -orient horizontal -tickinterval .2 \
	    -length 300
    pack $w.f.overall_gain -side top -fill x
}
