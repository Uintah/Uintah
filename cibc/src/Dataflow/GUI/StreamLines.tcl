#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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


itcl_class SCIRun_Visualization_StreamLines {
    inherit Module

    constructor {config} {
        set name StreamLines
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.e
	frame $w.e.labs
	frame $w.e.ents
	label $w.e.labs.tolerance -text "Error Tolerance" -just left
	entry $w.e.ents.tolerance -textvariable $this-tolerance
	label $w.e.labs.stepsize -text "Step Size" -just left
	entry $w.e.ents.stepsize -textvariable $this-stepsize
	label $w.e.labs.maxsteps -text "Maximum Steps" -just left
	entry $w.e.ents.maxsteps -textvariable $this-maxsteps
	label $w.e.labs.nthreads -text "Number of Threads" -just left
	iwidgets::spinint $w.e.ents.nthreads \
	    -range {1 256} -step 1 \
	    -textvariable $this-nthreads \
	    -width 10 -fixed 10 -justify left


	pack $w.e.labs.tolerance $w.e.labs.stepsize $w.e.labs.maxsteps \
	    $w.e.labs.nthreads -side top -anchor w
	pack $w.e.ents.tolerance $w.e.ents.stepsize $w.e.ents.maxsteps \
	    $w.e.ents.nthreads -side top -anchor w

	pack $w.e.labs $w.e.ents -side left


	frame $w.direction -relief groove -borderwidth 2
	label $w.direction.label -text "Direction"
	radiobutton $w.direction.neg -text "Negative" \
	    -variable $this-direction -value 0
	radiobutton $w.direction.both -text "Both" \
	    -variable $this-direction -value 1
	radiobutton $w.direction.pos -text "Positive" \
	    -variable $this-direction -value 2

	pack $w.direction.label -side top -fill both
	pack $w.direction.neg $w.direction.both $w.direction.pos \
	    -side left -fill both


	frame $w.value -relief groove -borderwidth 2
	label $w.value.label -text "Streamline Value"
	frame $w.value.left
	frame $w.value.middle
	frame $w.value.right
	radiobutton $w.value.left.value -text   "Seed Value" \
	    -variable $this-value -value 0
	label $w.value.left.blank -text ""
	radiobutton $w.value.middle.index -text "Seed Index" \
	    -variable $this-value -value 1
	radiobutton $w.value.middle.incr -text  "Integration Step" \
	    -variable $this-value -value 2
	radiobutton $w.value.right.delta -text  "Distance from Seed" \
	    -variable $this-value -value 3
	radiobutton $w.value.right.total -text  "Streamline Length" \
	    -variable $this-value -value 4

	pack $w.value.left.value $w.value.left.blank -side top -anchor w
	pack $w.value.middle.index $w.value.middle.incr -side top -anchor w
	pack $w.value.right.delta $w.value.right.total -side top -anchor w

	pack $w.value.label -side top -fill both
	pack $w.value.left $w.value.middle $w.value.right -side left -anchor w


	frame $w.meth -relief groove -borderwidth 2
	label $w.meth.label -text "Streamline Computation Method"
	radiobutton $w.meth.cw -text "Cell Walk" \
	    -variable $this-method -value 5
	radiobutton $w.meth.ab -text "Adams-Bashforth Multi-Step" \
	    -variable $this-method -value 0
	#radiobutton $w.meth.o2 -text "Adams Moulton Multi Step" \
	    -variable $this-method \
#	    -value 1
	radiobutton $w.meth.heun -text "Heun Method" \
	    -variable $this-method -value 2
	radiobutton $w.meth.rk4 -text "Classic 4th Order Runge-Kutta" \
	    -variable $this-method -value 3
	radiobutton $w.meth.rkf -text "Adaptive Runge-Kutta-Fehlberg" \
	    -variable $this-method -value 4
	
	pack $w.meth.label -side top -fill both
	pack $w.meth.cw $w.meth.ab $w.meth.heun $w.meth.rk4 $w.meth.rkf \
	    -side top -anchor w


	checkbutton $w.filter -text "Filter Colinear Points" \
		-variable $this-remove-colinear-pts -justify left

	pack $w.meth $w.e $w.direction $w.value $w.filter \
	    -side top -e y -f both -padx 5 -pady 5

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


