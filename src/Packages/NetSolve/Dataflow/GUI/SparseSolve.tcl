itcl_class NetSolve_Matrix_SparseSolve {
    
    inherit Module
    
    constructor {config} {
	set name SparseSolve
	initialize
    }

    method initialize {} {
	set $this-method PETSc
	set $this-maxiter 100
	set $this-target_error 0.0001
	set $this-final_error ---
	set $this-final_iterations ---
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	wm title $w "SparseSolve"

	frame $w.f
	pack $w.f

	frame $w.f.t
	frame $w.f.t2
	frame $w.f.b
	frame $w.f.b2
	frame $w.f.b3
	frame $w.f.b4
	frame $w.f.b5
	frame $w.f.b6
	frame $w.f.b7
	pack $w.f.t $w.f.t2 $w.f.b $w.f.b2 $w.f.b3\
	     $w.f.b4 $w.f.b5 $w.f.b6 $w.f.b7 -side top -f x -e y

	label $w.f.t.method -text "Method" -foreground darkblue
	pack $w.f.t.method -side top -f x -e y -pady 10

	frame $w.f.t2.l
	frame $w.f.t2.r
	pack $w.f.t2.l $w.f.t2.r -side left -f x -e y

	label $w.f.t2.l.iter_label -text "Iterative"
	label $w.f.t2.r.dir_label -text "Direct"
	pack $w.f.t2.l.iter_label $w.f.t2.r.dir_label -side top -f x

	frame $w.f.t2.l.t
	frame $w.f.t2.l.b
	frame $w.f.t2.r.t
	frame $w.f.t2.r.b
	pack $w.f.t2.l.t $w.f.t2.l.b $w.f.t2.r.t $w.f.t2.r.b -side top -f x

	radiobutton $w.f.t2.l.t.petsc -text "PETSc" -value PETSc\
		-variable $this-method
	radiobutton $w.f.t2.l.b.aztec -text "Aztec" -value Aztec\
		-variable $this-method
	pack $w.f.t2.l.t.petsc $w.f.t2.l.b.aztec -side left -f x -e y
	
	radiobutton $w.f.t2.r.t.superlu -text "SuperLU" -value SuperLU\
		-variable $this-method
	radiobutton $w.f.t2.r.b.ma28 -text "MA28" -value MA28\
		-variable $this-method
	pack $w.f.t2.r.t.superlu $w.f.t2.r.b.ma28 -side left -f x -e y

	label $w.f.b.start_label -text "Initial parameters"\
		-foreground darkblue
	pack $w.f.b.start_label -side top -f x -e y -pady 10

	label $w.f.b2.err_label -text "Error tollerance" -justify right
	entry $w.f.b2.err -textvariable $this-target_error
	pack $w.f.b2.err $w.f.b2.err_label -side right -padx 5 -pady 5\
		-f x -e y

	label $w.f.b3.iter_label -text "Max iterations" -justify right
	entry $w.f.b3.iter -textvariable $this-maxiter
	pack $w.f.b3.iter $w.f.b3.iter_label -side right -padx 5 -pady 5\
		-f x -e y
	
	label $w.f.b4.end_label -text "Final results"\
		-foreground darkblue
	pack $w.f.b4.end_label -side top -f x -e y -pady 10

	label $w.f.b5.err_label -text "Error achieved" -justify right
	entry $w.f.b5.err -textvariable $this-final_error -state disabled
	pack $w.f.b5.err $w.f.b5.err_label -side right -padx 5 -pady 5\
		-f x -e y

	label $w.f.b6.iter_label -text "Number iterations required"\
		-justify right
	entry $w.f.b6.iter -textvariable $this-final_iterations -state disabled
	pack $w.f.b6.iter $w.f.b6.iter_label -side right -padx 5 -pady 5\
		-f x -e y

	button $w.f.b7.execute -text "Execute" -command "$this-c needexecute"
	pack $w.f.b7.execute -pady 5
    }

    method show_results {error iterations} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	}

	$w.f.b5.err configure -state normal 
	$w.f.b5.err delete 0 end
	$w.f.b5.err insert 0 $error
	$w.f.b5.err configure -state disabled

	$w.f.b6.iter configure -state normal
	$w.f.b6.iter delete 0 end
	$w.f.b6.iter insert 0 $iterations
	$w.f.b6.iter configure -state disabled
    }
}