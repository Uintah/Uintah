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
#    File   : NrrdToField.tcl
#    Author : Darby Van Uitert
#    Date   : March 2004

catch {rename Teem_DataIO_NrrdToField ""}

itcl_class Teem_DataIO_NrrdToField {
    inherit Module
    constructor {config} {
        set name NrrdToField
        set_defaults
    }

    method set_defaults {} {
	global $this-permute
	global $this-build-eigens
	global $this-quad-or-tet
	global $this-struct-unstruct

	set $this-permute 0
	set $this-build-eigens 0
	set $this-quad-or-tet "Auto"
	set $this-struct-unstruct "Auto"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

	checkbutton $w.f.options.permute -text "Permute Data" \
	    -variable $this-permute
	pack $w.f.options.permute -side top  -expand yes -fill x
	
	checkbutton $w.f.options.eigens -text "Build Eigen decomposition for Tensor field" -variable $this-build-eigens
	pack $w.f.options.eigens -side top -expand yes -fill x

	iwidgets::labeledframe $w.f.options.quadtet \
	    -labelpos nw -labeltext "Unstructured Cell Type when Points per Connection equals 4:"
	pack $w.f.options.quadtet -side top -expand yes -fill x

	set quadtet [$w.f.options.quadtet childsite]

	radiobutton $quadtet.auto -text "Auto" \
	    -variable $this-quad-or-tet -value "Auto"
	radiobutton $quadtet.tet -text "Tet" \
	    -variable $this-quad-or-tet -value "Tet"
	radiobutton $quadtet.quad -text "Quad" \
	    -variable $this-quad-or-tet -value "Quad"

	pack $quadtet.auto $quadtet.tet $quadtet.quad  -side left -anchor nw -padx 3


	iwidgets::labeledframe $w.f.options.pccurve \
	    -labelpos nw -labeltext "Structured/Unstructured Ambiguity:"
	pack $w.f.options.pccurve -side top -expand yes -fill x
	set pccurve [$w.f.options.pccurve childsite]

	radiobutton $pccurve.auto -text "Auto" \
	    -variable $this-struct-unstruct -value "Auto"
	radiobutton $pccurve.pc -text "Point Cloud" \
	    -variable $this-struct-unstruct -value "PointCloud"
	radiobutton $pccurve.curve -text "Struct Curve" \
	    -variable $this-struct-unstruct -value "StructCurve"
	pack $pccurve.auto $pccurve.pc $pccurve.curve  -side left -anchor nw -padx 3

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


