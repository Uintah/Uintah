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

# GUI for PrismNrrdConverter module
# by Allen R. Sanderson
# May 2003

# This GUI interface is for selecting a file name via the makeOpenFilebox
# and other reading functions.

catch {rename Fusion_Fields_PrismNrrdConverter ""}

itcl_class Fusion_Fields_PrismNrrdConverter {
    inherit Module
    constructor {config} {
        set name PrismNrrdConverter
        set_defaults
    }

    method set_defaults {} {

	global $this-datasets
	set $this-datasets ""

	global $this-points
	set $this-points -1

	global $this-connect
	set $this-connect -1
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

	toplevel $w

	frame $w.grid
	label $w.grid.l -text "Points - Connection List: (Execute to show)" \
	    -width 45 -just left

	pack $w.grid.l -side left
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
		    $w.datasets.$i.l configure -text $dataset

		} else {
		    frame $w.datasets.$i

		    set len [string length $dataset]

		    radiobutton $w.datasets.$i.p -text "" -width 0 \
			-anchor w -just left -variable $this-points -value $i

		    radiobutton $w.datasets.$i.c -text "" -width 0 \
			-anchor w -just left -variable $this-connect -value $i

		    label $w.datasets.$i.l -text $dataset \
			-width $len -just left

		    pack $w.datasets.$i.p $w.datasets.$i.c $w.datasets.$i.l \
			-side left
		}

		pack $w.datasets.$i -side top

		incr i 1
	    }
	}
    }
}
