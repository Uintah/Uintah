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

#    File   : TendAnvol.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendAnvol ""}

itcl_class Teem_Tend_TendAnvol {
    inherit Module
    constructor {config} {
        set name TendAnvol
        set_defaults
    }
    method set_defaults {} {
        global $this-aniso_metric
        set $this-aniso_metric  "tenAniso_FA"

        global $this-threshold
        set $this-threshold 0.5


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


	make_labeled_radio $w.f.options.aniso_metric \
	    "Anisotropy Metric"  "" top  $this-aniso_metric \
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
		{"radius of root circle is 2*sqrt(Q/9)" "tenAniso_Q"}
		{"phase of root circle is acos(R/Q^3)" "tenAniso_R"}
		{"sqrt(Q^3 - R^2)" "tenAniso_S"}
		{"R/Q^3" "tenAniso_Th"}
		{"Zhukov's invariant-based anisotropy metric" "tenAniso_Cz"}
		{"plain old trace" "tenAniso_Tr"}}

        pack $w.f.options.aniso_metric -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.threshold -labeltext "threshold:" -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
	pack $w.f -expand 1 -fill x
    }
}
