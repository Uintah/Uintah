
proc uiFrameTest {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    frame $w.f
    pack $w.f -padx 2 -pady 2
    set n "$modid needexecute "

    button $w.f.button -text GO! -command $n
    pack $w.f.button -in $w.f -side left -padx 2 -pady 2 -anchor w
    fscale $w.f.slide -label Scale -from 0.0 -to 1000.0 -length 6c -showvalue true \
	    -orient horizontal -activeforeground SteelBlue2 -resolution .01 \
	    -digits 3 -variable widget_scale,$w -command $n
    pack $w.f.slide -in $w.f -side top -padx 2 -pady 2 -anchor w
}
