
proc uiMorphMesher3d {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
	raise $w
	return;
    }
    toplevel $w
    frame $w.f
    pack $w.f -padx 2 -pady 2

    frame $w.f.num_layers
    pack $w.f.num_layers
    global num_layers,$modid
    set num_layers,$modid 4
    label $w.f.num_layers.label -text "Number of mesh layers: "
    scale $w.f.num_layers.scale -variable num_layers,$modid \
        -from 1 -to 9 -command "$modid needexecute "\
        -showvalue true -tickinterval 1 \
        -orient horizontal
    pack $w.f.num_layers.label $w.f.num_layers.scale -fill x
}
