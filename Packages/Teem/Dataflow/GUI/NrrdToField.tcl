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

itcl_class Teem_DataIO_NrrdToField {
    inherit Module
    constructor {config} {
        set name NrrdToField

	global $this-build-eigens

        set_defaults
    }

    method set_defaults {} {
	set $this-build-eigens 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.c
	pack $w.c -side top -e y -f both -padx 5 -pady 5
	checkbutton $w.c.buildeigens -text \
	    "Build Eigendecomposition for Tensor Fields" -variable \
	    $this-build-eigens
	pack $w.c.buildeigens -side top -expand yes -fill x
    }
}


