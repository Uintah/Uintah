
itcl_class SolveMatrix {
    inherit Module
    constructor {config} {
	set name SolveMatrix
	set_defaults
    }
    method set_defaults {} {
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
		raise $w
		return;
	}

	toplevel $w
	wm minsize $w 200 20
	set n "$this-c needexecute "

	frame $w.iter
	pack $w.iter -side top -fill x
	label $w.iter.lab -text "Iteration: "
	pack $w.iter.lab -side left
	label $w.iter.val -text "0"
	pack $w.iter.val -side right


	frame $w.first
	pack $w.first -side top -fill x
	label $w.first.lab -text "Original Error: "
	pack $w.first.lab -side left
	label $w.first.val -text "9.9999e99"
	pack $w.first.val -side right

	frame $w.current
	pack $w.current -side top -fill x
	label $w.current.lab -text "Current Error: "
	pack $w.current.lab -side left
	label $w.current.val -text "9.9999e99"
	pack $w.current.val -side right

	frame $w.final
	pack $w.final -side top -fill x
	label $w.final.lab -text "Target Error: "
	pack $w.final.lab -side left
	label $w.final.val -text "9.9999e99"
	pack $w.final.val -side right

    }
    method update_iter {iter firsterr currenterr finalerr} {
	set w .ui$this
	if {[winfo exists $w]} {
	    $w.iter.val config -text $iter
	    $w.first.val config -text $firsterr
	    $w.current.val config -text $currenterr
	    $w.final.val config -text $finalerr
	    update idletasks
	}
    }
}


