proc combo_listbox {p {prompt ""} } {
    global $p
    global $p.entry
    global $p.listbox

    frame $p
    prompted_entry $p.entry $prompt
    iwidgets::scrolledlistbox $p.listbox -vscrollmode static \
        -hscrollmode dynamic -scrollmargin 3
    pack $p.entry -side top -fill x -expand true
    pack $p.listbox -side top -fill both -expand true
}


