
proc uiTYPEWriter {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    set filetype,$modid binary
    radiobutton $w.binary -text "Binary" -relief flat \
	    -variable filetype,$modid -value "binary"
    radiobutton $w.ascii -text "ASCII" -relief flat \
	    -variable filetype,$modid -value "text"
    $w.binary select
    pack $w.binary $w.ascii -side left
    entry $w.f -textvariable filename,$modid -width 40 \
	    -borderwidth 2 -relief sunken
    pack $w.f -side bottom
    bind $w.f <Return> "$modid needexecute "
}
