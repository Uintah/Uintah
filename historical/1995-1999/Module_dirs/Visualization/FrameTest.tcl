
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
    scale $w.f.slide -label Scale -from 0.0 -to 10.0 -length 6c -showvalue true \
	    -orient horizontal -resolution .001 \
	    -digits 8 -variable widget_scale,$modid -command $n
    pack $w.f.slide -in $w.f -side top -padx 2 -pady 2 -anchor w

    frame $w.f.wids
    radiobutton $w.f.wids.point -text PointWidget -variable widget_type,$modid -value 0 -command $n
    radiobutton $w.f.wids.arrow -text ArrowWidget -variable widget_type,$modid -value 1 -command $n
    radiobutton $w.f.wids.guage -text GuageWidget -variable widget_type,$modid -value 2 -command $n
    radiobutton $w.f.wids.frame -text FrameWidget -variable widget_type,$modid -value 3 -command $n
    radiobutton $w.f.wids.sframe -text ScaledFrameWidget -variable widget_type,$modid -value 4 -command $n
    radiobutton $w.f.wids.square -text SquareWidget -variable widget_type,$modid -value 5 -command $n
    radiobutton $w.f.wids.ssquare -text ScaledSquareWidget -variable widget_type,$modid -value 6 -command $n
    radiobutton $w.f.wids.box -text BoxWidget -variable widget_type,$modid -value 7 -command $n
    radiobutton $w.f.wids.sbox -text ScaledBoxWidget -variable widget_type,$modid -value 8 -command $n
    radiobutton $w.f.wids.cube -text CubeWidget -variable widget_type,$modid -value 9 -command $n
    radiobutton $w.f.wids.scube -text ScaledCubeWidget -variable widget_type,$modid -value 10 -command $n

    pack $w.f.wids.point $w.f.wids.arrow $w.f.wids.guage $w.f.wids.frame $w.f.wids.sframe $w.f.wids.square $w.f.wids.ssquare $w.f.wids.box $w.f.wids.sbox $w.f.wids.cube $w.f.wids.scube -in $w.f.wids -side top -padx 2 -pady 2 -anchor w
    pack $w.f.wids -in $w.f
}
