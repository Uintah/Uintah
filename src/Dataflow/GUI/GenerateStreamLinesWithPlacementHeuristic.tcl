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

itcl_class SCIRun_Visualization_GenerateStreamLinesWithPlacementHeuristic {
    inherit Module

    constructor {config} {
        set name GenerateStreamLinesWithPlacementHeuristic

	global $this-numsl;
	global $this-numpts;
	
	global $this-stepsize
	global $this-stepout
	global $this-maxsteps
	global $this-minmag
	global $this-direction
	global $this-method

	global $this-minx
	global $this-maxper
	global $this-ming
	global $this-maxg
	global $this-numsamples

        set_defaults
    }

    method set_defaults {} {
	set $this-numsl 10
	set $this-numpts 10

	set $this-stepsize 0.01
	set $this-stepout 100
	set $this-maxsteps 10000
	set $this-minmag 1e-7
	set $this-direction 1
	set $this-method 0

	set $this-minper 0
	set $this-maxper 1
	set $this-ming 0 
	set $this-maxg 1
	set $this-numsamples 3
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.a
	frame $w.a.labs
	frame $w.a.ents
	label $w.a.labs.numsl -text "Number of Streamlines" -just left
	entry $w.a.ents.numsl -textvariable $this-numsl
	label $w.a.labs.numpts -text "Number of Trials" -just left
	entry $w.a.ents.numpts -textvariable $this-numpts

	pack $w.a.labs.numsl $w.a.labs.numpts -side top -anchor w
	pack $w.a.ents.numsl $w.a.ents.numpts -side top -anchor e
	pack $w.a.labs $w.a.ents -side left -expand yes -fill x

	frame $w.s
	frame $w.s.labs
	frame $w.s.ents
	label $w.s.labs.stepsize -text "Stepsize" -just left
	entry $w.s.ents.stepsize -textvariable $this-stepsize
	label $w.s.labs.stepout -text "Step Output" -just left
	entry $w.s.ents.stepout -textvariable $this-stepout
	label $w.s.labs.maxsteps -text "Maximum Steps" -just left
	entry $w.s.ents.maxsteps -textvariable $this-maxsteps
	label $w.s.labs.minmag -text "Minimal Magnitude" -just left
	entry $w.s.ents.minmag -textvariable $this-minmag

	pack $w.s.labs.stepsize $w.s.labs.stepout $w.s.labs.maxsteps $w.s.labs.minmag \
	    -side top -anchor w
	pack $w.s.ents.stepsize $w.s.ents.stepout $w.s.ents.maxsteps $w.s.ents.minmag \
	    -side top -anchor e
	pack $w.s.labs $w.s.ents -side left -expand yes -fill x

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

	frame $w.meth -relief groove -borderwidth 2
	label $w.meth.label -text "Streamline Integration Method"
	radiobutton $w.meth.euler -text "Euler" \
	    -variable $this-method -value 0
	radiobutton $w.meth.rk2 -text "Runge-Kutta 2nd Order" \
	    -variable $this-method -value 1
	radiobutton $w.meth.rk4 -text "Runge-Kutta 4th Order" \
	    -variable $this-method -value 2
	
	pack $w.meth.label -side top -fill both
	pack $w.meth.euler $w.meth.rk2 $w.meth.rk4 \
	    -side top -anchor w


	frame $w.e -relief groove -borderwidth 2
	label $w.e.label -text "Render Function Parameters"
	frame $w.e.labs 
	frame $w.e.ents
	label $w.e.labs.minper -text "Minimal Radius \[%\]" -just left
	entry $w.e.ents.minper -textvariable $this-minper
	label $w.e.labs.maxper -text "Maximal Radius \[%\]" -just left
	entry $w.e.ents.maxper -textvariable $this-maxper
	label $w.e.labs.ming -text "Minimal Magnitude" -just left
	entry $w.e.ents.ming -textvariable $this-ming
	label $w.e.labs.maxg -text "Maximal Magnitude" -just left
	entry $w.e.ents.maxg -textvariable $this-maxg
	label $w.e.labs.numsamples -text "Number of Samples" -just left
	entry $w.e.ents.numsamples -textvariable $this-numsamples

	pack $w.e.label -side top -fill both
	pack $w.e.labs.minper $w.e.labs.maxper $w.e.labs.ming $w.e.labs.maxg \
	    $w.e.labs.numsamples -side top -anchor w
	pack $w.e.ents.minper $w.e.ents.maxper $w.e.ents.ming $w.e.ents.maxg \
	    $w.e.ents.numsamples -side top -anchor e
	pack $w.e.labs $w.e.ents -side left -expand yes -fill x

	pack $w.a $w.s $w.direction $w.meth $w.e -side top -expand yes -fill x  \
		-padx 5 -pady 5

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
