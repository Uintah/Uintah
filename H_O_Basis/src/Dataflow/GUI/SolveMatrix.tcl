#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


catch {rename SolveMatrix ""}

itcl_class SCIRun_Math_SolveMatrix {
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
	global $this-emit_partial $this-emit_iter
        global $this-graphs
	
        set $this-target_error 0.001
	set $this-method "Conjugate Gradient & Precond. (SCI)"
        set $this-precond jacobi
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
	set $this-emit_iter 50
	set $this-np 4
        set $this-graphs {}
    }
    
    
    method switchmethod {} {
	global $this-method
	set w .ui[modname]
	set meth [set $this-method]
	if {($meth == "Conjugate Gradient & Precond. (SCI)") || \
	    ($meth == "BiConjugate Gradient & Precond. (SCI)") || \
	    ($meth == "Jacobi & Precond. (SCI)")} {
	    pack forget $w.stat                  
	    pack $w.converg $w.graph -side top -padx 2 -pady 2 -fill x
	    $w.np.label configure -state normal
	    $w.np.entry configure -state normal
	    #foreach t [winfo children $w.precond] {
		#if {[winfo class $t] == "Radiobutton"} {
		 #   $t configure -state disabled
		#}
	    #}
        } else {
#	    pack forget $w.graph $w.converg
	    pack $w.stat  -side top -padx 2 -pady 2 -fill x
	    $w.np.label configure -state disabled
	    $w.np.entry configure -state disabled
	    #foreach t [winfo children $w.precond] {
		#if {[winfo class $t] == "Radiobutton"} {
		    #$t configure -state normal
		#}
	    #}
	    
        }
    }
    
    
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	
	toplevel $w
	wm minsize $w 300 20
	set n "$this-c needexecute "
	
	frame $w.np
	pack $w.np -side top
	
	label $w.np.label -text "Number of threads"
	entry $w.np.entry -width 2 -textvariable $this-np 
	
	pack $w.np.label $w.np.entry -side left -anchor w
	
	frame $w.m
	label $w.m.method_label -text "Current Method: " -anchor w \
                                -width 20
	label $w.m.method -textvar $this-method -width 40 -anchor w \
                          -fg darkred
	frame $w.p
	label $w.p.precon_label -text "Current Preconditioner: " -anchor w \
                                -width 20
        label $w.p.precond -textvar $this-precond -width 40 -anchor w \
                           -fg darkred
        pack $w.m $w.p -side top
        pack $w.m.method_label $w.m.method $w.p.precon_label $w.p.precond \
             -side left 

	iwidgets::tabnotebook  $w.tabs -raiseselect true -tabpos n\
                               -width 400 -height 150
        pack $w.tabs -side top -fill x
	set methods [$w.tabs add -label "Methods"]
	set precons [$w.tabs add -label "Preconditioners (PETSc only)"]

        $w.tabs view 0

        iwidgets::scrolledframe $methods.f -hscrollmode none \
                                -vscrollmode dynamic
        iwidgets::scrolledframe $precons.f -hscrollmode none \
                                -vscrollmode dynamic

        pack $methods.f $precons.f -f both -e y

	set meth [$methods.f childsite]
	set prec [$precons.f childsite]

	make_labeled_radio $meth.f "" "$this switchmethod" \
		top $this-method\
		{{"Conjugate Gradient & Precond. (SCI)" "Conjugate Gradient & Precond. (SCI)"}\
		{"BiConjugate Gradient & Precond. (SCI)" "BiConjugate Gradient & Precond. (SCI)"}\
		{"Jacobi & Precond. (SCI)" "Jacobi & Precond. (SCI)"}
	        {"KSPRICHARDSON (PETSc)" "KSPRICHARDSON (PETSc)"}
		{"KSPCHEBYCHEV (PETSc)" "KSPCHEBYCHEV (PETSc)"}
		{"KSPCG (PETSc)" "KSPCG (PETSc)"}
		{"KSPGMRES (PETSc)" "KSPGMRES (PETSc)"}
		{"KSPTCQMR (PETSc)" "KSPTCQMR (PETSc)"}
		{"KSPBCGS (PETSc)" "KSPBCGS (PETSc)"}
		{"KSPCGS (PETSc)" "KSPCGS (PETSc)"}
		{"KSPTFQMR (PETSc)" "KSPTFQMR (PETSc)"}
		{"KSPCR (PETSc)" "KSPCR (PETSc)"}
		{"KSPLSQR (PETSc)" "KSPLSQR (PETSc)"}
		{"KSPBICG (PETSc)" "KSPBICG (PETSc)"}
		{"KSPPREONLY (PETSc)" "KSPPREONLY (PETSc)"}}
		   
	
	make_labeled_radio $prec.f "" ""\
                top $this-precond\
                {{jacobi jacobi}
		{bjacobi bjacobi}
                {sor sor}
		{eisenstat eisenstat}
		{icc icc}
		{ilu ilu}
		{asm asm}
		{sles sles}
		{lu lu}
	        {mg mg}
	        {spai spai}
	        {milu milu}
	        {nn nn}
		{cholesky cholesky}
                {ramg ramg}
		{none none}}

	if { [$this-c petscenabled] == 0 } {
            for {set i 3} {$i<15} {incr i} {
	     $meth.f.$i configure -state disabled
	    }
	    for {set i 0} {$i<16} {incr i} {
	     $prec.f.$i configure -state disabled
            }
        }

        
	pack $meth.f -side top -fill x -pady 2
	pack $prec.f -side top -fill x -pady 2
	
	global $this-target_error
	expscale $w.target_error -orient horizontal -label "Target error:" \
		-variable $this-target_error -command ""

	pack $w.target_error -side top -fill x -pady 2

	scale $w.maxiter -orient horizontal -label "Maximum Iterations:" \
		-variable $this-maxiter -from 0 -to 20000
	pack $w.maxiter -side top -fill x -pady 2

	checkbutton $w.emit -variable $this-emit_partial \
		-text "Emit partial solutions"
	pack $w.emit -side top -fill x -pady 2

	scale $w.emititer -orient horizontal \
		-label "Partial Solution Emitted Every:" \
		-variable $this-emit_iter -from 0 -to 2000
	pack $w.emititer -side top -fill x -pady 2
	
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
#	pack $w.converg.flopcount -side top -fill x
	label $w.converg.flopcount.lab -text "Flop Count: "
	pack $w.converg.flopcount.lab -side left
	label $w.converg.flopcount.val -textvariable $this-flops
	pack $w.converg.flopcount.val -side right
	
	frame $w.converg.floprate
#	pack $w.converg.floprate -side top -fill x
	label $w.converg.floprate.lab -text "MFlops: "
	pack $w.converg.floprate.lab -side left
	label $w.converg.floprate.val -textvariable $this-floprate
	pack $w.converg.floprate.val -side right
	
	frame $w.converg.memcount
#	pack $w.converg.memcount -side top -fill x
	label $w.converg.memcount.lab -text "Memory bytes accessed: "
	pack $w.converg.memcount.lab -side left
	label $w.converg.memcount.val -textvariable $this-memrefs
	pack $w.converg.memcount.val -side right
	
	frame $w.converg.memrate
#	pack $w.converg.memrate -side top -fill x
	label $w.converg.memrate.lab -text "Memory bandwidth (MB/sec):"
	pack $w.converg.memrate.lab -side left
	label $w.converg.memrate.val -textvariable $this-memrate
	pack $w.converg.memrate.val -side right
	
	global $this-target_error
	set err [set $this-target_error]
	
	blt::graph $w.graph -title "Convergence" -height 250 \
		-plotbackground gray99
	$w.graph yaxis configure -logscale true -title "Error (RMS)"  -min [expr $err/10] -max 1 -loose true
	$w.graph xaxis configure -title "Iteration" \
		-loose true
	bind $w.graph <ButtonPress-1> "$this select_error $w.graph %x %y"
	bind $w.graph <Button1-Motion> "$this move_error $w.graph %x %y"
	bind $w.graph <ButtonRelease-1> "$this deselect_error $w.graph %x %y"
	set iter 1
	$w.graph element create "Current Target" -linewidth 0
	$w.graph element configure "Current Target" -data "0 $err" \
		-symbol diamond
	pack $w.graph -fill x
        add_graph $w.graph

	reset_graph
	switchmethod

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    
    protected error_selected false
    protected tmp_error
    
    method select_error {graph wx wy} {
	global $this-target_error $this-iteration
	set err [set $this-target_error]
	set iter [set $this-iteration]
	set errpos [$graph transform $iter $err]
	set erry [lindex $errpos 1]
	set errx [lindex $errpos 0]
	if {abs($wy-$erry) < 5} {
            foreach g [set $this-graphs] {
                $g element configure "Current Target" -color yellow
            }
	    set error_selected true
	}
    }
    
    method move_error {graph wx wy} {
	global $this-target_error
	if {$error_selected == "true"} {
	    set newerror [lindex [$graph invtransform $wx $wy] 1]
            foreach g [set $this-graphs] {
                $g element configure "Current Target" -ydata $newerror
            }
	}
    }
    
    method deselect_error {graph wx wy} {
	if {$error_selected == "true"} {
            foreach g [set $this-graphs] {
                $g element configure "Current Target" -color blue
            }
	    set error_selected false
	    set newerror [lindex [$graph invtransform $wx $wy] 1]
	    global $this-target_error
	    set $this-target_error $newerror
	}
    }
    
    protected min_error

    method add_graph {graph} {
        global $this-graphs
        set $this-graphs [concat [set $this-graphs] $graph]
    }

    method reset_graph_once {graph} {
	if {![winfo exists $graph]} {
	    return
	}
	catch "$graph element delete {Target Error}"
	catch "$graph element delete {Current Error}"

	$graph element create "Target Error" -linewidth 2 -color blue -symbol ""
	$graph element create "Current Error" -linewidth 2 -color red -symbol ""
	global $this-target_error
	set err [set $this-target_error]
	set iter 1
	$graph element configure "Target Error" -data "0 $err $iter $err"
	set min_error $err
	$graph yaxis configure -min [expr $err/10] -max 1
	$graph element configure "Current Target" -data "0 $err"
    }
    
    method reset_graph {} {
        foreach graph [set $this-graphs] {
            reset_graph_once $graph
        }
    }

    method append_graph_once {graph iter values errvalues} {
	if {![winfo exists $graph]} {
	    return
	}
	if {$values != ""} {
	    set xvals [$graph element cget "Current Error" -xdata]
	    set yvals [$graph element cget "Current Error" -ydata]
	    
	    for {set i 0} {$i<[llength $values]} {incr i} {
		lappend xvals [lindex $values $i]
		incr i
		lappend yvals [lindex $values $i]
	    }
	    $graph element configure "Current Error" -xdata $xvals
	    $graph element configure "Current Error" -ydata $yvals
	}
	if {$errvalues != ""} {
	    set xvals [$graph element cget "Target Error" -xdata]
	    set yvals [$graph element cget "Target Error" -ydata]
	    
	    for {set i 0} {$i<[llength $errvalues]} {incr i} {
		lappend xvals [lindex $errvalues $i]
		incr i
		lappend yvals [lindex $errvalues $i]
	    }
	    $graph element configure "Target Error" -xdata $xvals
	    $graph element configure "Target Error" -ydata $yvals
	}
	global $this-target_error
	set err [set $this-target_error]
	if {$err < $min_error} {
	    set min_error $err
	}
	$graph yaxis configure -min [expr $min_error/10]
	$graph element configure "Current Target" -xdata [expr $iter - 1]
    }

    method append_graph {iter values errvalues} {
        foreach graph [set $this-graphs] {
            append_graph_once $graph $iter $values $errvalues
        }
    }

    method finish_graph_once {graph} {
	if {![winfo exists $graph]} {
	    return
	}
	$graph element configure "Current Error" -color green
    }        

    method finish_graph {} {
        foreach graph [set $this-graphs] {
            finish_graph_once $graph
        }
    }
}
