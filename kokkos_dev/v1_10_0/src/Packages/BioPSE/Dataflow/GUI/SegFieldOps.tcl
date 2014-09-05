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

itcl_class BioPSE_Modeling_SegFieldOps {
    inherit Module
    constructor {config} {
        set name SegFieldOps
        set_defaults
    }

    method set_defaults {} {
	global $this-min_comp_size
	set $this-min_comp_size 100
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.s
	scale $w.s.s -orient horizontal -from 1 -to 1000 \
	    -showvalue true -variable "$this-min_comp_size"
	pack $w.s.s -side top -fill x -expand 1
	
	frame $w.b
	button $w.b.e -text "Execute" -command "$this-c needexecute"
	button $w.b.a -text "Audit" -command "$this-c audit"
	button $w.b.p -text "Print" -command "$this-c print"
	button $w.b.k -text "Absorb" -command "$this-c absorb"
	button $w.b.r -text "Reset" -command "$this-c reset"
	pack $w.b.e $w.b.a $w.b.p $w.b.p $w.b.k $w.b.r -side left -fill x -expand 1 -padx 3
	pack $w.s $w.b -side top -fill x -expand 1
    }
}
