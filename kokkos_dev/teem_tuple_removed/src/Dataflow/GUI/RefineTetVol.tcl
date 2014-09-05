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
#    File   : RefineTetVol.tcl
#    Author : Martin Cole
#    Date   : Thu Nov 13 10:10:00 2003

itcl_class SCIRun_FieldsCreate_RefineTetVol {
    inherit Module
    constructor {config} {
        set name RefineTetVol
        set_defaults
    }

    method set_defaults {} {
	global $this-cell_index
	global $this-execution_mode

	set $this-cell_index -1
	set $this-execution_mode sub_one
    }



    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f

	make_labeled_radio $w.f.rb \
	    "Execution mode" "" top \
	    $this-execution_mode \
	    {{"Subdivide to level:" sub_all} {"Subdivide cell index:" sub_one}}


	label $w.f.lab -text "Enter cell index or level"
	entry $w.f.cell_index -width 40 -textvariable $this-cell_index

     	pack $w.f.rb $w.f.lab $w.f.cell_index -side top -anchor w
	

	frame $w.controls
	button $w.controls.exc -text "Execute" -command "$this-c needexecute"
	button $w.controls.cancel -text "Cancel" -command "destroy $w"
	pack $w.controls.exc $w.controls.cancel -side left -fill both
	
	pack $w.f $w.controls -side top -expand yes -fill both -padx 5 -pady 5
    }
}




