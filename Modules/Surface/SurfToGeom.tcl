
proc uiSurfToGeom {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    wm minsize $w 100 100
    frame $w.f
    pack $w.f -padx 2 -pady 2 -fill x -expand yes
    set n "$modid needexecute "

    global range_min,$modid
    fscale $w.f.min -variable range_min,$modid -digits 4 \
	    -from 0.0 -to 90 -label "Color min:" \
	    -resolution .01 -showvalue true \
	    -orient horizontal
    fscale $w.f.max -variable range_max,$modid -digits 4 \
	    -from 0.0 -to 90 -label "Color max:" \
	    -resolution .01 -showvalue true \
	    -orient horizontal
    pack $w.f.min $w.f.max -side top -fill x
}
