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
	set $this-newtype "Same"
        set $this-oldtype "Unknown"
    }
    method make_entry {w text v} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v -state disabled
        pack $w.e -side right
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 20
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        global $this-oldtype
	make_entry $w.f.o "Old type: " $this-oldtype
        global $this-newtype
        make_labeled_radio $w.f.n "Cast to:" "" \
                top $this-newtype \
		{{"Same (pass-through)" Same} \
		{ColumnMatrix ColumnMatrix} \
		{DenseMatrix DenseMatrix} \
		{SparseRowMatrix SparseRowMatrix}}
	
	pack $w.f.o $w.f.n -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
