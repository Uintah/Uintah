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
 #  NrrdToMatrix.tcl
 #  Written by:
 #   Darby Van Uitert
 #   April 2004
 ##

itcl_class Teem_DataIO_NrrdToMatrix {
    inherit Module
    constructor {config} {
        set name NrrdToMatrix
        set_defaults
    }

    method set_defaults {} {
	global $this-nnz

	set $this-nnz 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	frame $w.nonzero
	pack $w.nonzero -side top -anchor nw -padx 3 -pady 3
	label $w.nonzero.l -text "Number of non-zero entries (for Sparse Matrix):"
	entry $w.nonzero.e -textvariable $this-nnz
	pack $w.nonzero.l $w.nonzero.e -side left -anchor nw -padx 3 -pady 3

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


