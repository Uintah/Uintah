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
#    File   : TendFiber.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendFiber ""}

itcl_class Teem_Tend_TendFiber {
    inherit Module
    constructor {config} {
        set name TendFiber
        set_defaults
    }
    method set_defaults {} {
        global $this-fibertype
        set $this-fibertype "tensorline"

        global $this-puncture
        set $this-puncture 0.0

        global $this-neighborhood
        set $this-neighborhood 2.0

	global $this-stepsize
	set $this-stepsize 0.01

	global $this-integration
	set $this-integration "Euler"

	global $this-use-aniso
	set $this-use-aniso 1

	global $this-aniso-metric
	set $this-aniso-metric "tenAniso_Cl2"

	global $this-aniso-thresh
	set $this-aniso-thresh 0.4

	global $this-use-length
	set $this-use-length 1
	
	global $this-length
	set $this-length 1

	global $this-use-steps
	set $this-use-steps 0
	
	global $this-steps
	set $this-steps 200

	global $this-use-conf
	set $this-use-conf 1

	global $this-conf-thresh
	set $this-conf-thresh 0.5

	global $this-kernel
	set $this-kernel "tent"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes -fill both
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	frame $w.f.options.type -borderwidth 2 -relief raised
	label $w.f.options.type.l -text "Fiber Algorithm"
	pack $w.f.options.type.l -side top -fill x -expand 1
	make_labeled_radio $w.f.options.type.alg \
	    "" "" \
	    left $this-fibertype \
	    {{"Major eigenvector" evec1} \
		 {"Tensorline (TL)" tensorline} \
		 {"Oriented tensors (OT)" zhukov}}
	pack $w.f.options.type.alg -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.type.punc -labeltext \
	    "  Puncture weighting (TL and OT):" -textvariable $this-puncture
        pack $w.f.options.type.punc -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.type.nbrhd -labeltext \
	    "  Neighborhood (OT):" -textvariable $this-neighborhood
        pack $w.f.options.type.nbrhd -side top -expand yes -fill x	
        iwidgets::entryfield $w.f.options.type.stepsize -labeltext \
	    "  Step size:" -textvariable $this-stepsize
	pack $w.f.options.type.stepsize -side top -fill x -expand 1
	make_labeled_radio $w.f.options.type.int \
	    "  Integration method" "" left $this-integration \
	    {{"Euler" "Euler"}
		{"RK4" "RK4"}}
	pack $w.f.options.type.int -side top -fill x -expand 1
	pack $w.f.options.type -side top -expand yes -fill x

	frame $w.f.options.filter -borderwidth 2 -relief raised
	label $w.f.options.filter.l -text "Kernel"
	make_labeled_radio $w.f.options.filter.kernel \
	    "" "" \
	    left $this-kernel \
	    {{"Box" box} \
		 {"Tent" tent} \
		 {"Cubic (Catmull-Rom)" cubicCR} \
		 {"Cubic (B-spline)" cubicBS} \
		 {"Quartic" quartic} \
		 {"Gaussian" gaussian}}
	pack $w.f.options.filter.l $w.f.options.filter.kernel -side top \
	    -expand yes -fill x
	pack $w.f.options.filter -side top -expand yes -fill x

	frame $w.f.options.stop -borderwidth 2 -relief raised

	label $w.f.options.stop.l -text "Stopping Criteria"
	pack $w.f.options.stop.l -side top

	frame $w.f.options.stop.len -borderwidth 2 -relief groove
	checkbutton $w.f.options.stop.len.use -text "Fiber length:" \
	    -variable $this-use-length
        iwidgets::entryfield $w.f.options.stop.len.length -labeltext \
	    "" -textvariable $this-length
	pack $w.f.options.stop.len.use -side left
	pack $w.f.options.stop.len.length -side left -fill x -expand 1
	pack $w.f.options.stop.len -side top -fill x -expand 1
	
	frame $w.f.options.stop.steps -borderwidth 2 -relief groove
	checkbutton $w.f.options.stop.steps.use -text \
	    "Number of steps:" -variable $this-use-steps
        iwidgets::entryfield $w.f.options.stop.steps.num -labeltext \
	    "" -textvariable $this-steps
	pack $w.f.options.stop.steps.use -side left -fill x
	pack $w.f.options.stop.steps.num -side left -fill x -expand 1
	pack $w.f.options.stop.steps -side top -fill x -expand 1
	
	frame $w.f.options.stop.conf -borderwidth 2 -relief groove
	checkbutton $w.f.options.stop.conf.use -text \
	    "Confidence threshold:" -variable $this-use-conf
        iwidgets::entryfield $w.f.options.stop.conf.thresh -labeltext \
	    "" -textvariable $this-conf-thresh
	pack $w.f.options.stop.conf.use -side left -fill x
	pack $w.f.options.stop.conf.thresh -side left -fill x -expand 1
	pack $w.f.options.stop.conf -side top -fill x -expand 1
	pack $w.f.options.stop -side top -fill x -expand 1

	frame $w.f.options.stop.a -borderwidth 2 -relief groove
	checkbutton $w.f.options.stop.a.use -text "Anisotropy threshold:" \
	    -variable $this-use-aniso
	frame $w.f.options.stop.a.f
	pack $w.f.options.stop.a.use -anchor nw
	pack $w.f.options.stop.a.f -side top -fill x -expand 1
	label $w.f.options.stop.a.f.l -text "    "
	make_labeled_radio $w.f.options.stop.a.f.aniso \
	    ""  "" top  $this-aniso-metric \
	    {{"Westin's linear (first version)" "tenAniso_Cl1"} 
		{"Westin's planar (first version)" "tenAniso_Cp1"} 
		{"Westin's linear + planar (first version)" "tenAniso_Ca1"}
		{"Westin's spherical (first version)" "tenAniso_Cs1"}
		{"gk's anisotropy type (first version)" "tenAniso_Ct1"}
		{"Westin's linear (second version)" "tenAniso_Cl2"}
		{"Westin's planar (second version)" "tenAniso_Cp2"}
		{"Westin's linear + planar (second version)" "tenAniso_Ca2"}
		{"Westin's spherical (second version)" "tenAniso_Cs2"}
		{"gk's anisotropy type (second version)" "tenAniso_Ct2"}
		{"Bass+Pier's relative anisotropy" "tenAniso_RA"}
		{"(Bass+Pier's fractional anisotropy)/sqrt(2)" "tenAniso_FA"}
		{"volume fraction = 1-(Bass+Pier's volume ratio)" "tenAniso_VF"}
		{"plain old trace" "tenAniso_Tr"}}
	pack $w.f.options.stop.a.f.l $w.f.options.stop.a.f.aniso \
	    -side left
        iwidgets::entryfield $w.f.options.stop.a.thresh -labeltext \
	    "  Threshold value:" -textvariable $this-aniso-thresh
	pack $w.f.options.stop.a.thresh -side top -fill x
	pack $w.f.options.stop.a -side top -fill x -expand 1

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
