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
#    File   : FieldToNrrd.tcl
#    Author : Martin Cole
#    Date   : Thu Jan 16 09:44:07 2003

itcl_class Teem_DataIO_FieldToNrrd {
    inherit Module
    constructor {config} {
        set name FieldToNrrd
        set_defaults
    }

    method set_defaults {} {
	global $this-label
	set $this-label "unknown"
    }
   
    # do not allow spaces in the label
    method valid_string {ind str} {
	set char "a"
	
	set char [string index $str $ind]
	if {$ind >= 0 && [string equal $char " "]} {
	    return 0
	}
	return 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.lab -relief groove -borderwidth 2

	pack $w.lab -side top -e y -f both -padx 5 -pady 5
	
	label $w.lab.label -text "Label Data   "
	entry $w.lab.dat-label -textvariable $this-label \
	    -validate key -validatecommand "$this valid_string %i %P"

	pack $w.lab.label $w.lab.dat-label -side left

	button $w.execute -text "Set Label" -command "destroy $w"
	pack $w.execute -side top -e n -f both
    }
}


