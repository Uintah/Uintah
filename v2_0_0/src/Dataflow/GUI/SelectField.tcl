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

itcl_class SCIRun_FieldsOther_SelectField {
    inherit Module
    constructor {config} {
        set name SelectField

	global $this-stampvalue
	global $this-runmode

        set_defaults
    }

    method set_defaults {} {
	set $this-stampvalue 100
	set $this-runmode 0
	# 0 nothing 1 accumulate 2 replace
    }

    method replace {} {
	set $this-runmode 2
	$this-c needexecute
    }

    method accumulate {} {
	set $this-runmode 1
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row1
	frame $w.row3
	frame $w.row4

	pack $w.row1 $w.row3 $w.row4 -side top -e y -f both -padx 5 -pady 5
	
	label $w.row1.value_label -text "Selection Value"
	entry $w.row1.value -textvariable $this-stampvalue
	pack $w.row1.value_label $w.row1.value -side left

	button $w.row3.execute -text "Replace" -command "$this replace"
	pack $w.row3.execute -side top -e n -f both

	button $w.row4.execute -text "Accumulate" -command "$this accumulate"
	pack $w.row4.execute -side top -e n -f both
    }
}


