
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
    global widget_scale,$modid
    set widget_scale,$modid .01
    fscale $w.f.slide -label Scale -from 0.0 -to 10.0 -length 6c -showvalue true \
	    -orient horizontal -activeforeground SteelBlue2 -resolution .001 \
	    -digits 8 -variable widget_scale,$modid -command $n
    pack $w.f.slide -in $w.f -side top -padx 2 -pady 2 -anchor w

    frame $w.f.wids
    radiobutton $w.f.wids.arrow -text ArrowWidget -variable widget_type,$modid -value 0
    radiobutton $w.f.wids.frame -text FrameWidget -variable widget_type,$modid -value 1
    pack $w.f.wids.arrow $w.f.wids.frame -in $w.f.wids -side left -padx 2 -pady 2 -anchor w
    pack $w.f.wids -in $w.f
}
