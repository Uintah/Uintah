
proc uiTYPEReader {modid} {
    set w .ui$modid
    if {[winfo exists $w]} {
        raise $w
        return;
    }
    toplevel $w
    makeFilebox $w filename,$modid "$modid needexecute" "destroy $w"
}
