
proc uiIsoSurface {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    frame $w.f 
    pack $w.f -padx 2 -pady 2
    set n "$modid needexecute "
    
    button $w.f.emit_surface -text "Emit Surface" -command "emitIsoSurface $modid"
    pack $w.f.emit_surface
}

proc emitIsoSurface {modid} {
    global emit_surface,$modid
    set emit_surface,$modid 1
    $modid needexecute
}
