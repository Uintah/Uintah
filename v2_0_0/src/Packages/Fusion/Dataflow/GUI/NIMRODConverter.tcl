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

# GUI for NIMRODConverter module
# by Allen R. Sanderson
# May 2003

# This GUI interface is for selecting a file name via the makeOpenFilebox
# and other reading functions.

catch {rename Fusion_Fields_NIMRODConverter ""}

itcl_class Fusion_Fields_NIMRODConverter {
    inherit Module
    constructor {config} {
        set name NIMRODConverter
        set_defaults
    }

    method set_defaults {} {

	global $this-datasets
	set $this-datasets ""

	global $this-nmodes
	set $this-nmodes 0

	global $this-mode
	set $this-mode 0
    }

    method ui {} {
	global env

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

#	if {[winfo exists $w]} {
#	    set child [lindex [winfo children $w] 0]
#
#	    # $w withdrawn by $child's procedures
#	    raise $child
#	    return;
#	}

	# When building the UI prevent the selection from taking place
	# since it is not valid.

	toplevel $w

	frame $w.modes

	label $w.modes.label -text "Mode" -width 6 -anchor w -just left
	pack $w.modes.label -side left	    

	set_modes [set $this-nmodes]

	pack $w.modes -side top -pady 5

	frame $w.grid
	label $w.grid.l -text "Inputs: (Execute to show list)" -width 30 -just left

	pack $w.grid.l  -side left
	pack $w.grid -side top

	frame $w.datasets
	
	global $this-datasets
	set_names [set $this-datasets]

	pack $w.datasets -side top -pady 10

	frame $w.misc
	button $w.misc.execute -text "Execute" -command "$this-c needexecute"
	button $w.misc.close -text Close -command "destroy $w"
	pack $w.misc.execute $w.misc.close -side left -padx 25

	pack $w.misc -side bottom -pady 10
    }

    method set_modes {nnodes} {

        set w .ui[modname]

	if [ expr [winfo exists $w] ] {
	    pack forget $w.modes.label

	    for {set i 0} {$i < 10} {incr i 1} {
		if [ expr [winfo exists $w.modes.$i] ] {
		    pack forget $w.modes.$i
		}
	    }

	    if { $nnodes > 0 } {

		global $this-nmodes
		global $this-mode
		set $this-nmodes $nnodes
		set $this-mode $nnodes

		pack $w.modes.label -side left

		for {set i 0} {$i <= $nnodes} {incr i 1} {
		    if [ expr [winfo exists $w.modes.$i] ] {
			$w.modes.$i.label configure -text "$i" -width 2
		    } else {
			frame $w.modes.$i
			
			label $w.modes.$i.label -text "$i" \
			    -width 2 -anchor w -just left
			radiobutton $w.modes.$i.button \
			    -variable $this-mode -value $i
			
			pack $w.modes.$i.label $w.modes.$i.button -side left
			pack $w.modes.$i.label -side left
			
		    }

		    pack $w.modes.$i -side left
		}
		
		$w.modes.$nnodes.label configure -text "Sum" -width 4
	    }
	}
    }

    method set_names {datasets} {

	global $this-datasets
	set $this-datasets $datasets

        set w .ui[modname]

	if [ expr [winfo exists $w] ] {

	    for {set i 0} {$i < 10} {incr i 1} {
		if [ expr [winfo exists $w.datasets.$i] ] {
		    pack forget $w.datasets.$i
		}
	    }

	    set i 0

	    foreach dataset $datasets {

		if [ expr [winfo exists $w.datasets.$i] ] {
		    $w.datasets.$i configure -text $dataset
		} else {
		    set len [expr [string length $dataset] + 5 ]
		    label $w.datasets.$i -text $dataset -width $len \
			-anchor w -just left
		}

		pack $w.datasets.$i -side top

		incr i 1
	    }
	}
    }
}
