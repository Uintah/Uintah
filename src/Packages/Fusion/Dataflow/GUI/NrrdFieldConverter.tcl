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

# GUI for NrrdFieldConverter module
# by Allen R. Sanderson
# May 2003

catch {rename Fusion_Fields_NrrdFieldConverter ""}

itcl_class Fusion_Fields_NrrdFieldConverter {
    inherit Module
    constructor {config} {
        set name NrrdFieldConverter
        set_defaults
    }

    method set_defaults {} {

	global $this-datasets
	set $this-datasets ""

	global $this-permute
	set $this-permute 0
 
	global $this-nomesh
	set $this-nomesh 0
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

	# Permute option
	global $this-permute

	frame $w.permute
	label $w.permute.label -text "Permute the data" \
	    -width 40 -anchor w -just left
	checkbutton $w.permute.button -variable $this-permute
	
	pack $w.permute.button $w.permute.label -side left
	pack $w.permute -side top -pady 5


	# Nomesh option
	global $this-nomesh

	frame $w.mesh
	label $w.mesh.label -text "No Mesh - regular topology and geometry" \
	    -width 40 -anchor w -just left
	checkbutton $w.mesh.button -variable $this-nomesh
	
	pack $w.mesh.button $w.mesh.label -side left
	pack $w.mesh -side top -pady 5


	# Input dataset label
	frame $w.label
	label $w.label.l -text "Inputs: (Execute to show list)" -width 30 \
	    -just left

	pack $w.label.l  -side left
	pack $w.label -side top -pady 5


	# Input Dataset
	frame $w.datasets
	pack $w.datasets -side top -pady 5

	global $this-datasets
	set_names [set $this-datasets]


	makeSciButtonPanel $w $w $this
	moveToCursor $w
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
