#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

itcl_class SCIRun_Visualization_StreamLines {
    inherit Module
    constructor {config} {
        set name StreamLines

	global $this-stepsize
	global $this-tolerance
	global $this-maxsteps
	global $this-direction
	global $this-color
	global $this-remove-colinear
	global $this-method
	global $this-np

        set_defaults
    }

    method set_defaults {} {
	set $this-tolerance 0.0001
	set $this-stepsize 0.01
	set $this-maxsteps 2000
	set $this-direction 1
	set $this-color 1
	set $this-remove-colinear 1
	set $this-method 4
	set $this-np 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
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
	entry $w.e.ents.nthreads -textvariable $this-np

	pack $w.e.labs.tolerance $w.e.labs.stepsize $w.e.labs.maxsteps \
	    $w.e.labs.nthreads -side top -anchor w
	pack $w.e.ents.tolerance $w.e.ents.stepsize $w.e.ents.maxsteps \
	    $w.e.ents.nthreads -side top -anchor e
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


	frame $w.color -relief groove -borderwidth 2
	label $w.color.label -text "Color Style"
	radiobutton $w.color.const -text "Constant" -variable $this-color \
		-value 0
	radiobutton $w.color.incr -text "Increment" -variable $this-color \
		-value 1

	pack $w.color.label -side top -fill both
	pack $w.color.const $w.color.incr -side top -anchor w


	frame $w.meth -relief groove -borderwidth 2
	label $w.meth.label -text "Computation Method"
	radiobutton $w.meth.ab -text "Adams-Bashforth Multi-Step" \
	    -variable $this-method -value 0
	#radiobutton $w.meth.o2 -text "Adams Moulton Multi Step" -variable $this-method \
        #	    -value 1
	radiobutton $w.meth.heun -text "Heun Method" \
	    -variable $this-method -value 2
	radiobutton $w.meth.rk4 -text "Classic 4th Order Runge-Kutta" \
	    -variable $this-method -value 3
	radiobutton $w.meth.rkf -text "Adaptive Runge-Kutta-Fehlberg" \
	    -variable $this-method -value 4
	
	pack $w.meth.label -side top -fill both
	pack $w.meth.ab $w.meth.heun $w.meth.rk4 $w.meth.rkf \
	    -side top -anchor w


	checkbutton $w.filter -text "Filter Colinear Points" \
		-variable $this-remove-colinear -justify left


	frame $w.row4
	button $w.row4.execute -text "Execute" -command "$this-c needexecute"
	
	pack $w.row4.execute -side top -e n -f both

	pack $w.meth $w.e $w.direction $w.color $w.filter $w.row4 \
		-side top -e y -f both \
		-padx 5 -pady 5
    }
}


