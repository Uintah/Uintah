
proc makeFilebox {w var command} {
    frame $w.f
    label $w.f.title -text "Filename: "
    entry $w.f.filename -relief sunken -width 40 -textvariable $var
    bind $w.f.filename <Return> "$command"
    pack $w.f.title -side left -padx 2 -pady 2
    pack $w.f.filename -side left -padx 2 -pady 2
    pack $w.f
}

