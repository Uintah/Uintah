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

itcl_class SCIRun_Visualization_StreamLines {
    inherit Module
    constructor {config} {
        set name StreamLines

	global $this-stepsize
	global $this-tolerance
	global $this-maxsteps

        set_defaults
    }

    method set_defaults {} {
	set $this-tolerance 0.0001
	set $this-stepsize 0.0001
	set $this-maxsteps 2000
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row1
	frame $w.row2
	frame $w.row3
	frame $w.row4

	pack $w.row1 $w.row2 $w.row3 $w.row4 -side top -e y -f both \
		-padx 5 -pady 5
	
	label $w.row1.tolerance_label -text "Error Tolerance"
	entry $w.row1.tolerance -textvariable $this-tolerance
	label $w.row2.stepsize_label -text "Step Size"
	entry $w.row2.stepsize -textvariable $this-stepsize
	label $w.row3.maxsteps_label -text "Maximum Steps"
	entry $w.row3.maxsteps -textvariable $this-maxsteps

	pack $w.row1.tolerance_label $w.row1.tolerance -side left
	pack $w.row2.stepsize_label $w.row2.stepsize -side left
	pack $w.row3.maxsteps_label $w.row3.maxsteps -side left

	button $w.row4.execute -text "Execute" -command "$this-c execute"
	
	pack $w.row4.execute -side top -e n -f both
    }
}


