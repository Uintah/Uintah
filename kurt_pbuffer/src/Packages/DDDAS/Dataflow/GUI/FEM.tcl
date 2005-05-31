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

itcl_class DDDAS_PDESolver_FEM {
    inherit Module
    constructor {config} {
        set name FEM
        set_defaults
    }
    method set_defaults {} {
        global $this-method
        global $this-nu
        global $this-poly_degree
        global $this-iter_method
        global $this-print_iter
        global $this-max_iter
        global $this-restart_iter
        global $this-rtol
	set $this-method 2
	set $this-nu 1
        set $this-poly_degree 1
        set $this-iter_method 2
	set $this-print_iter 1
        set $this-max_iter 1000
        set $this-restart_iter 50
        set $this-rtol 1E-8
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	make_labeled_radio $w.method "Method" "" top $this-method \
	    {{"Std FEM" 1} {"BO" 2} {"NIPG" 3} {"IP" 4}}
	pack $w.method -anchor w

	frame $w.nu
	label $w.nu.l -text "Nu: "
	entry $w.nu.e -textvar $this-nu -width 8
	pack $w.nu.l $w.nu.e -side left -anchor w
	pack $w.nu -side top -anchor w

	frame $w.frm1 -height 2 -relief sunken -bd 1
	pack $w.frm1 -fill x

	make_labeled_radio $w.poly_degree "Polynomial Degree" "" top \
	    $this-poly_degree {{"Linear" 1} {"Quadratic" 2}}
	pack $w.poly_degree -anchor w

	frame $w.frm2 -height 2 -relief sunken -bd 1
	pack $w.frm2 -fill x

	make_labeled_radio $w.iter_method "Iterative Method" "" top \
	    $this-iter_method {{"PCG" 1} {"GMRES" 2} {"BiCGSTAB" 3}}
	checkbutton $w.print_iter -text "Print Iterations" \
	    -variable $this-print_iter
	pack $w.iter_method $w.print_iter -anchor w

	frame $w.max_iter
	label $w.max_iter.l -text "Maximum Iterations: "
	entry $w.max_iter.e -textvar $this-max_iter -width 8
	pack $w.max_iter.l $w.max_iter.e -side left -anchor w
	pack $w.max_iter -side top -anchor w

	frame $w.restart_iter
	label $w.restart_iter.l -text "GMRES Restart Iterations: "
	entry $w.restart_iter.e -textvar $this-restart_iter -width 6
	pack $w.restart_iter.l $w.restart_iter.e -side left -anchor w
	pack $w.restart_iter -side top -anchor w

	frame $w.rtol
	label $w.rtol.l -text "Relative Tolerance: "
	entry $w.rtol.e -textvar $this-rtol -width 8
	pack $w.rtol.l $w.rtol.e -side left -anchor w
	pack $w.rtol -side top -anchor w

	makeSciButtonPanel $w $w $this
	moveToCursor $w
     }
}
