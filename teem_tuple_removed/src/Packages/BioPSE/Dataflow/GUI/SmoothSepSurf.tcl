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
