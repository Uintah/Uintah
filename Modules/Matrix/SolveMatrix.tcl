
itcl_class SolveMatrix {
    inherit Module
    constructor {config} {
	set name SolveMatrix
	set_defaults
    }
    method set_defaults {} {
	global $this-target_error $this-method $this-orig_error
	global $this-current_error $this-flops $this-floprate $this-iteration
	set $this-target_error 1.e-4
	set $this-method conjugate_gradient
	set $this-orig_error 9.99999e99
	set $this-current_error 9.99999e99
	set $this-flops 0
	set $this-floprate 0
	set $this-iteration 0
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
		raise $w
		return;
	}

	toplevel $w
	wm minsize $w 300 20
	set n "$this-c needexecute "

	make_labeled_radio $w.method "Solution Method" $n \
		top $this-method \
		{{"Jacobi" jacobi} \
		{"Conjugate Gradient" conjugate_gradient}}

	pack $w.method -side top -fill x -pady 2

	expscale $w.target_error -orient horizontal -label "Target error:" \
		-variable $this-target_error -command $n
	pack $w.target_error -side top -fill x -pady 2


	frame $w.converg -borderwidth 2 -relief ridge
	pack $w.converg -side top -padx 2 -pady 2 -fill x

	frame $w.converg.iter
	pack $w.converg.iter -side top -fill x
	label $w.converg.iter.lab -text "Iteration: "
	pack $w.converg.iter.lab -side left
	label $w.converg.iter.val -textvariable $this-iteration
	pack $w.converg.iter.val -side right

	frame $w.converg.first
	pack $w.converg.first -side top -fill x
	label $w.converg.first.lab -text "Original Error: "
	pack $w.converg.first.lab -side left
	label $w.converg.first.val -textvariable $this-orig_error
	pack $w.converg.first.val -side right

	frame $w.converg.current
	pack $w.converg.current -side top -fill x
	label $w.converg.current.lab -text "Current Error: "
	pack $w.converg.current.lab -side left
	label $w.converg.current.val -textvariable $this-current_error
	pack $w.converg.current.val -side right

	frame $w.converg.flopcount
	pack $w.converg.flopcount -side top -fill x
	label $w.converg.flopcount.lab -text "Flop Count: "
	pack $w.converg.flopcount.lab -side left
	label $w.converg.flopcount.val -textvariable $this-flops
	pack $w.converg.flopcount.val -side right

	frame $w.converg.floprate
	pack $w.converg.floprate -side top -fill x
	label $w.converg.floprate.lab -text "MFlops: "
	pack $w.converg.floprate.lab -side left
	label $w.converg.floprate.val -textvariable $this-floprate
	pack $w.converg.floprate.val -side right

	global $this-target_error
	set err [set $this-target_error]

	blt_graph $w.graph -title "Convergence" -height 250 \
		-plotbackground gray50
	$w.graph yaxis configure -logscale true -title "error (RMS)"
	$w.graph xaxis configure -title "Iteration" \
		-loose true

	set iter 1
	$w.graph element create "Target Error" -linewidth 1
	$w.graph element configure "Target Error" -data "0 $err $iter $err"

	pack $w.graph -fill x
    }
    method reset_graph {} {
	set w .ui$this
	if {![winfo exists $w]} {
	    return
	}
	catch "$w.graph element delete {Target Error}"
	catch "$w.graph element delete {Current Error}"
	$w.graph element create "Target Error" -linewidth 0 -foreground blue
	$w.graph element create "Current Error" -linewidth 0 -foreground red
	global $this-target_error
	set err [set $this-target_error]
	set iter 1
	$w.graph element configure "Target Error" -data "0 $err $iter $err"
    }

    method append_graph {iter values} {
	set w .ui$this
	if {![winfo exists $w]} {
	    return
	}
	if {$values != ""} {
	    $w.graph element append "Current Error" "$values"
	}
	global $this-target_error
	set err [set $this-target_error]
	$w.graph yaxis configure -min [expr $err/10]
	$w.graph element configure "Target Error" -data "0 $err $iter $err"
    }

    method finish_graph {} {
	set w .ui$this
	$w.graph element configure "Current Error" -foreground green
    }
}
