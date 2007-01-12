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

catch {rename Butson_Modeling_AppendSparse ""}

itcl_class Butson_Modeling_AppendSparse {
    inherit Module
    constructor {config} {
        set name AppendSparse
        set_defaults
    }

    method set_defaults {} {
	global $this-appendmode
	set $this-appendmode "rows"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

	frame $w.location -relief groove -borderwidth 2
	radiobutton $w.location.row -text "Append Rows" \
	    -variable $this-appendmode -value rows
	radiobutton $w.location.col -text "Append Columns" \
	    -variable $this-appendmode -value columns

	pack $w.location.row $w.location.col \
	    -side top -anchor w

	pack $w.location -side top -fill x -expand 1 \
	    -padx 5 -pady 5
    }
}
