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

itcl_class SCIRun_Math_CastMatrix {
    inherit Module
    constructor {config} {
        set name CastMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-newtype
        global $this-oldtype
	global $this-nrow
	global $this-ncol
	set $this-newtype "Same"
        set $this-oldtype "Unknown"
	set $this-nrow "??"
	set $this-ncol "??"
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 20

	frame $w.i -relief ridge -borderwidth 2
	pack $w.i -side top -padx 2 -pady 2 -side top

	frame $w.i.size
        pack $w.i.size -side top
	global $this-nrow $this-ncol
	label $w.i.size.label -text "Input Matrix Size:   nrows ="
	label $w.i.size.row -textvariable $this-nrow
	label $w.i.size.x -text "  ncols ="
	label $w.i.size.col -textvariable $this-ncol
	pack $w.i.size.label $w.i.size.row $w.i.size.x $w.i.size.col -side left

        frame $w.i.type 
	pack $w.i.type -side top
        global $this-oldtype
	label $w.i.type.l -text "Input Matrix Type: " 
	label $w.i.type.v -textvariable $this-oldtype
	global $this-space
	label $w.i.type.s -textvariable $this-space
	pack $w.i.type.l $w.i.type.v $w.i.type.s -side left
	
	frame $w.otype -relief ridge -borderwidth 2
        pack $w.otype -side top -expand yes -padx 2 -pady 2 -fill x
        global $this-newtype
        make_labeled_radio $w.otype.r \
		"Output Matrix Type" "" \
                top $this-newtype \
		{{"Same (pass-through)" Same} \
		{ColumnMatrix ColumnMatrix} \
		{DenseMatrix DenseMatrix} \
		{SparseRowMatrix SparseRowMatrix}}
	pack $w.otype.r -side top -expand 1 -fill x
    }
}
