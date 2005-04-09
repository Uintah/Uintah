
proc old_uiDelaunay {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    wm minsize $w 400 10
    frame $w.f 
    pack $w.f -padx 2 -pady 2 -fill both
    set n "$modid needexecute "

    fscale $w.f.maxnode -variable maxnode,$modid -from 0 -to 1400 \
	    -showvalue true -tickinterval 200 \
	    -orient horizontal
    pack $w.f.maxnode -fill x -side top
    button $w.f.go -text "Go" -command $n
    pack $w.f.go
}
