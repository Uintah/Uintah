catch {rename SolveMatrix ""}

itcl_class Dataflow_Matrix_SolveMatrix {
    inherit Module
    constructor {config} {
	set name SolveMatrix
	set_defaults
    }
    method set_defaults {} {
	global $this-target_error $this-method_sci $this-method $this-precond $this-orig_error
	global $this-current_error $this-flops $this-floprate $this-iteration
	global $this-memrefs $this-memrate $this-maxiter
	global $this-use_previous_so
	global $this-np

        set $this-target_error 1.e-4
	set $this-method conjugate_gradient_sci
        set $this-precond Diag_P
	set $this-orig_error 0
	set $this-current_error 0
	set $this-flops 0
	set $this-floprate 0
	set $this-memrefs 0
	set $this-memrate 0
	set $this-iteration 0
	set $this-maxiter 200
	set $this-use_previous_soln 1
	set $this-emit_partial 1
	set $this-np 4
    }
 

 method switchmethod {} {
        global $this-method
        set w .ui[modname]
        set meth [set $this-method]
     if {($meth == "conjugate_gradient_sci") || ($meth == "jacoby_sci") || ($meth == "bi_conjugate_gradient_sci")} {
           pack forget $w.stat                  
	   pack $w.converg $w.graph -side top -padx 2 -pady 2 -fill x
	 foreach t [winfo children $w.precond] {
	     if {[winfo class $t] == "Radiobutton"} {
		 $t configure -state disabled
	     }
	 }
        } else {
	     pack forget $w.graph $w.converg
             pack $w.stat  -side top -padx 2 -pady 2 -fill x
	 foreach t [winfo children $w.precond] {
	     if {[winfo class $t] == "Radiobutton"} {
		 $t configure -state normal
	     }
	 }

        }
    }
 

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
		raise $w
		return;
	}

	toplevel $w
	wm minsize $w 300 20
	set n "$this-c needexecute "

	button $w.execute -text "Execute" -command $n
	pack $w.execute -side top -fill x -pady 2 -padx 2
        
	frame $w.np
	pack $w.np -side top

	label $w.np.label -text "Number of threads"
	entry $w.np.entry -width 2 -relief flat -textvariable $this-np 
	    
	pack $w.np.label $w.np.entry -side left -anchor w

	make_labeled_radio $w.method "Solution Method" "$this switchmethod" \
		top $this-method\
            {{"Conjugate Gradient" conjugate_gradient }\
             {"Conjugate Gradient Squared Iteration" conj_grad_squared}\
	         {"BiConjugate Gradient Iteration" bi_conjugate_gradient}\
             {"BiConjugate Gradient Iteration Stabilized" bi_conjugate_gradient_stab}\
             {"Quasi Minimal Residual Iteration" quasi_minimal_res}\
             {"Generalized Minimum Residual Iteration" gen_min_res_iter}\
        	 {"Richardson Iterations" richardson_iter}\
			 {"Conjugate Gradient & Precond. (Dataflow)" conjugate_gradient_sci}\
             {"BiConjugate Gradient & Precond. (Dataflow)" bi_conjugate_gradient_sci}\
	         {"Jacoby & Precond. (Dataflow)" jacoby_sci}}


                 
             

        
    make_labeled_radio $w.precond "Preconditioner" ""\
        top $this-precond\
		{{"DiagPreconditioner" Diag_P}\
		{"IC Preconditioner" IC_P}\
	    {"ILU Preconditioner" ILU_P}}
    

    pack $w.method -side top -fill x -pady 2

	pack $w.precond -side top -fill x -pady 2

#	expscale $w.target_error -orient horizontal -label "Target error:" \
#		-variable $this-target_error -command ""
#	pack $w.target_error -side top -fill x -pady 2

	scale $w.maxiter -orient horizontal -label "Maximum Iterations:" \
		-variable $this-maxiter -from 0 -to 400
	pack $w.maxiter -side top -fill x -pady 2

	checkbutton $w.use_prev -variable $this-use_previous_soln \
		-text "Use previous solution as initial guess"
	pack $w.use_prev -side top -fill x -pady 2

        frame $w.stat -borderwidth 2 -relief ridge
        pack $w.stat  -side top -padx 2 -pady 2 -fill x

	frame $w.converg -borderwidth 2 -relief ridge
	pack $w.converg -side top -padx 2 -pady 2 -fill x

        frame $w.stat.status
	pack $w.stat.status -side top -fill x
	label $w.stat.status.lab -text "Job Status: "
	pack $w.stat.status.lab -side left
	label $w.stat.status.val -textvariable $this-status
	pack $w.stat.status.val -side right
      
        frame $w.stat.iter
	pack $w.stat.iter -side top -fill x
	label $w.stat.iter.lab -text "Total Iterations: "
	pack $w.stat.iter.lab -side left
	label $w.stat.iter.val -textvariable $this-iteration
	pack $w.stat.iter.val -side right

        frame $w.stat.current
	pack $w.stat.current -side top -fill x
	label $w.stat.current.lab -text "Tolerance Achieved: "
	pack $w.stat.current.lab -side left
	label $w.stat.current.val -textvariable $this-current_error
	pack $w.stat.current.val -side right
	
	
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

	frame $w.converg.memcount
	pack $w.converg.memcount -side top -fill x
	label $w.converg.memcount.lab -text "Memory bytes accessed: "
	pack $w.converg.memcount.lab -side left
	label $w.converg.memcount.val -textvariable $this-memrefs
	pack $w.converg.memcount.val -side right

	frame $w.converg.memrate
	pack $w.converg.memrate -side top -fill x
	label $w.converg.memrate.lab -text "Memory bandwidth (MB/sec):"
	pack $w.converg.memrate.lab -side left
	label $w.converg.memrate.val -textvariable $this-memrate
	pack $w.converg.memrate.val -side right

	global $this-target_error
	set err [set $this-target_error]


        expscale $w.graph -orient horizontal \
		-variable $this-target_error -label "Target Error"
        pack $w.graph -fill x

#	blt_graph $w.graph -title "Convergence" -height 250 \
#		-plotbackground gray99
#	$w.graph yaxis configure -logscale true -title "error (RMS)"
#	$w.graph xaxis configure -title "Iteration" \
#		-loose true
#	bind $w.graph <ButtonPress-1> "$this select_error %x %y"
#	bind $w.graph <Button1-Motion> "$this move_error %x %y"
#	bind $w.graph <ButtonRelease-1> "$this deselect_error %x %y"
#	set iter 1
#	$w.graph element create "Current Target" -linewidth 2
#	$w.graph element configure "Current Target" -data "0 $err" \
#		-symbol diamond
#	pack $w.graph -fill x
    switchmethod
    }

    protected error_selected false
    protected tmp_error

    method select_error {wx wy} {
#	global $this-target_error $this-iteration
#	set w .ui[modname]
#	set err [set $this-target_error]
#	set iter [set $this-iteration]
#	set errpos [$w.graph transform $iter $err]
#	set erry [lindex $errpos 1]
#	set errx [lindex $errpos 0]
#	if {abs($wy-$erry)+abs($wx-$errx) < 5} {
#	    $w.graph element configure "Current Target" -foreground yellow
#	    set error_selected true
#	}
    }

    method move_error {wx wy} {
#	set w .ui[modname]
#	set newerror [lindex [$w.graph invtransform $wx $wy] 1]
#	$w.graph element configure "Current Target" -ydata $newerror
    }

    method deselect_error {wx wy} {
#	set w .ui[modname]
#	$w.graph element configure "Current Target" -foreground blue
#	set error_selected false
#	set newerror [lindex [$w.graph invtransform $wx $wy] 1]
#	global $this-target_error
#	set $this-target_error $newerror
    }

    protected min_error

    method reset_graph {} {
#	set w .ui[modname]
#	if {![winfo exists $w]} {
#	    return
#	}
#	catch "$w.graph element delete {Target Error}"
#	catch "$w.graph element delete {Current Error}"
#	$w.graph element create "Target Error" -linewidth 2 -foreground blue
#	$w.graph element create "Current Error" -linewidth 2 -foreground red 
#	global $this-target_error
#	set err [set $this-target_error]
#	set iter 1
#	$w.graph element configure "Target Error" -data "0 $err $iter $err"
#	set min_error $err
#	$w.graph element configure "Current Target" -data "0 $err"
    }

    method append_graph {iter values errvalues} {
#	set w .ui[modname]
#	if {![winfo exists $w]} {
#	    return
#	}
#	if {$values != ""} {
#	    $w.graph element append "Current Error" "$values"
#	}
#	if {$errvalues != ""} {
#	    $w.graph element append "Target Error" "$errvalues"
#	}
#	global $this-target_error
#	set err [set $this-target_error]
#	if {$err < $min_error} {
#	    set min_error $err
#	}
#	$w.graph yaxis configure -min [expr $min_error/10]
#	$w.graph element configure "Current Target" -xdata $iter
    }

    method finish_graph {} {
#	set w .ui[modname]
#	$w.graph element configure "Current Error" -foreground green
    }
}












