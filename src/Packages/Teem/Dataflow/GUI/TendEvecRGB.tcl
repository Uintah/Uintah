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
#    File   : TendEvecRGB.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendEvecRGB ""}

itcl_class Teem_Tend_TendEvecRGB {
    inherit Module
    constructor {config} {
        set name TendEvecRGB
        set_defaults
    }
    method set_defaults {} {
        global $this-evec
        set $this-evec "major"

        global $this-aniso_metric
        set $this-aniso_metric  "tenAniso_FA"

        global $this-background
        set $this-background 0.0

	global $this-gray
	set $this-gray 0.0

	global $this-gamma
	set $this-gamma 1.0
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

	frame $w.f.options.f1 -borderwidth 2 -relief groove
	make_labeled_radio $w.f.options.f1.evec \
	    "Eigenvector to use" "" top $this-evec \
	    {{"Major" "major"}
		{"Medium" "medium"}
		{"Minor" "minor"}}
	pack $w.f.options.f1.evec -side top -expand yes -fill x
	pack $w.f.options.f1 -side top -expand yes -fill x

	frame $w.f.options.f2 -borderwidth 2 -relief groove
	make_labeled_radio $w.f.options.f2.aniso_metric \
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
		{"plain old trace" "tenAniso_Tr"}}
	pack $w.f.options.f2.aniso_metric -side top -expand yes -fill x
	pack $w.f.options.f2 -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.background -labeltext \
	    "Background:" -textvariable $this-background
        pack $w.f.options.background -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.gray -labeltext \
	    "Gray:" -textvariable $this-gray
        pack $w.f.options.gray -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.gamma -labeltext \
	    "Gamma:" -textvariable $this-gamma
        pack $w.f.options.gamma -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
