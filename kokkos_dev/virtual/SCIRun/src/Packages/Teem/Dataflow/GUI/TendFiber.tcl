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

#    File   : TendFiber.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendFiber ""}

itcl_class Teem_Tend_TendFiber {
    inherit Module

    constructor {config} {
        set name TendFiber
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
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

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
