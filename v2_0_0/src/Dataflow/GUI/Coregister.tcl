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


catch {rename Coregister ""}

package require Iwidgets 3.0   

itcl_class SCIRun_FieldsOther_Coregister {
    inherit Module
    
    constructor {config} {
	set name Coregister
	set_defaults
    }
    
    method set_defaults {} {
	global $this-allowScale
	global $this-allowRotate
	global $this-allowTranslate
	global $this-seed
	global $this-iters
	global $this-misfitTol
	global $this-method
	set $this-allowScale 1
	set $this-allowRotate 1
	set $this-allowTranslate 1
	set $this-seed 1
	set $this-iters 1000
	set $this-misfitTol 0.001
	set $this-method "Analytic"
	trace variable $this-method w "$this switch_to_method"

    }

    method switch_to_method {name1 name2 op} {
	#puts stdout "switching"
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set mf [$window.f.meth childsite]
	    $mf.tabs view [set $this-method]
	}
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "
	
	checkbutton $w.f.s -text "Allow scale" \
		-variable $this-allowScale
	checkbutton $w.f.r -text "Allow rotate" \
		-variable $this-allowRotate
	checkbutton $w.f.t -text "Allow translate" \
		-variable $this-allowTranslate
	
	button $w.f.go -text "Execute" -command $n

	pack $w.f.s $w.f.r $w.f.t $w.f.go

	#  Methods
	iwidgets::labeledframe $w.f.meth -labelpos nw -labeltext "Methods"
	set mf [$w.f.meth childsite]
	
	iwidgets::tabnotebook  $mf.tabs -raiseselect true 
	#-fill both
	pack $mf.tabs -side top

	#  Method:

	set alg [$mf.tabs add -label "Analytic" \
		-command "$this select-alg 0"]
	
	set alg [$mf.tabs add -label "Procrustes" \
		-command "$this select-alg 1"]

	set alg [$mf.tabs add -label "Simplex" \
		-command "$this select-alg 2"]

	frame $alg.s
	pack $alg.s -side top
	label $alg.s.l -text "Seed: "
	entry $alg.s.e -textvariable $this-seed
	bind $alg.s.e <KeyPress-Return> $n
	pack $alg.s.l $alg.s.e -side left -fill x -expand 1

	frame $alg.iters
	pack $alg.iters -side top
	label $alg.iters.l -text "Max iterations: "
	entry $alg.iters.e -textvariable $this-iters
	bind $alg.iters.e <KeyPress-Return> $n
	pack $alg.iters.l $alg.iters.e -side left -fill x -expand 1

	frame $alg.misfit
	pack $alg.misfit -side top
	label $alg.misfit.l -text "Misfit tolerance: "
	entry $alg.misfit.e -textvariable $this-misfitTol
	bind $alg.misfit.e <KeyPress-Return> $n
	pack $alg.misfit.l $alg.misfit.e -side left -fill x -expand 1

	button $alg.b -text "Abort" -command "$this-c stop"
	pack $alg.b -side top

	$mf.tabs view [set $this-method]
	$mf.tabs configure -tabpos "n"
	
	pack $mf.tabs -side top
	pack $w.f.meth -side top
    }

    method select-alg { alg } {
	global $this-method

	if { $alg == 0 } {
	    set $this-method "Analytic"
	} elseif { $alg == 1 } {
	    set $this-method "Procrustes"
	} else {
	    set $this-method "Simplex"
	}
    }

}
