# an example field manipulation insertable UI

proc fm_ui_Rescale { p modname } {
    frame $p.l1
    frame $p.l2
    frame $p.l3
    frame $p.l4
    pack $p.l1 $p.l2 $p.l3 $p.l4 -padx 2 -pady 2
    
    label $p.l1.l -text "x factor: " -width 15
    entry $p.l1.factor -width 30
    label $p.l2.l -text "y factor: " -width 15
    entry $p.l2.factor -width 30
    label $p.l3.l -text "z factor: " -width 15
    entry $p.l3.factor -width 30
    pack $p.l1.l $p.l1.factor \
	 $p.l2.l $p.l2.factor \
	 $p.l3.l $p.l3.factor -side left -padx 2 -pady 2
}