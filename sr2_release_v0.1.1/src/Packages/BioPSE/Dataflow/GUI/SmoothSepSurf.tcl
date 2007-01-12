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


##
 #  SmoothSepSurf.tcl: Smooth the separating surfaces using Taubins's algorithm
 #
 #  Written by:
 #   David Weinstein
 #   School of Computing
 #   University of Utah
 #   March 2003
 #
 #  Copyright (C) 2003 SCI Institute
 # 
 ##

catch {rename BioPSE_Modeling_SmoothSepSurf ""}

itcl_class BioPSE_Modeling_SmoothSepSurf {
    inherit Module
    constructor {config} {
	set name SmoothSepSurf
	set_defaults
    }
    method set_defaults {} {
	global $this-N
	global $this-pb
	global $this-gamma
	global $this-constraintTCL
	global $this-constrainedTCL
	global $this-jitterTCL
	set $this-pb .4
	set $this-gamma .8
	set $this-N 10
	set $this-constrainedTCL 0
	set $this-constraintTCL 1
	set $this-jitterTCL 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"
	
	global $this-N
	global $this-pb
	global $this-gamma
	global $this-constrainedTCL
	global $this-constraintTCL
	global $this-jitterTCL
	scale $w.f.n -orient horizontal -label "Iterations: " \
		-variable $this-N -showvalue true -from 1 -to 100
	scale $w.f.pb -orient horizontal -label "Pass Band (k) : " \
		-variable $this-pb -resolution .01 -showvalue true \
		-from 0 -to 1
	scale $w.f.gamma -orient horizontal -label "Gamma: " \
		-variable $this-gamma -resolution .01 -showvalue true \
		-from 0 -to 1
	scale $w.f.constr -orient horizontal -label "Constraint: " \
		-variable $this-constraintTCL -resolution .1 -showvalue true \
		-from 1 -to 3
	frame $w.f.b
	button $w.f.b.r -text "Reset" -command "$this-c reset"
	button $w.f.b.e -text "Execute" -command "$this-c tcl_exec"
	button $w.f.b.p -text "Print" -command "$this-c print"
	checkbutton $w.f.b.c -text "Constrained" -variable $this-constrainedTCL
	checkbutton $w.f.b.j -text "Jitter" -variable $this-jitterTCL
	pack $w.f.b.r $w.f.b.e $w.f.b.p $w.f.b.c $w.f.b.j -side left -padx 4 -expand 1
	pack $w.f.n $w.f.pb $w.f.gamma $w.f.constr $w.f.b -side top -fill x -expand 1
	$this set_defaults
    }
}
