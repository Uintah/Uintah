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

itcl_class SCIRun_Fields_Probe {
    inherit Module
    constructor {config} {
        set name Probe
        set_defaults
    }

    method set_defaults {} {
	global $this-value
	global $this-locx
	global $this-locy
	global $this-locz
	global $this-node
	global $this-edge
	global $this-face
	global $this-cell
	global $this-moveto

	set $this-value ""
	set $this-locx ""
	set $this-locy ""
	set $this-locz ""
	set $this-node ""
	set $this-edge ""
	set $this-face ""
	set $this-cell ""
	set $this-moveto ""
    }

    method move_location {} {
	set $this-moveto "location"
	$this-c needexecute
    }

    method move_node {} {
	set $this-moveto "node"
	$this-c needexecute
    }

    method move_edge {} {
	set $this-moveto "edge"
	$this-c needexecute
    }

    method move_face {} {
	set $this-moveto "face"
	$this-c needexecute
    }

    method move_cell {} {
	set $this-moveto "cell"
	$this-c needexecute
    }

    method move_center {} {
	set $this-moveto "center"
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f
	frame $w.f.labels
	frame $w.f.entries
	frame $w.f.entries.loc


	label $w.f.labels.location -text "Location" -just left
	entry $w.f.entries.loc.locx -width 10 -textvariable $this-locx
	entry $w.f.entries.loc.locy -width 10 -textvariable $this-locy
	entry $w.f.entries.loc.locz -width 10 -textvariable $this-locz

	label $w.f.labels.value -text "Value" -just left
	entry $w.f.entries.value -width 40 -state disabled -textvariable $this-value

	label $w.f.labels.node -text "Node" -just left
	entry $w.f.entries.node -width 10 -textvariable $this-node

	label $w.f.labels.edge -text "Edge" -just left
	entry $w.f.entries.edge -width 10 -textvariable $this-edge

	label $w.f.labels.face -text "Face" -just left
	entry $w.f.entries.face -width 10 -textvariable $this-face

	label $w.f.labels.cell -text "Cell" -just left
	entry $w.f.entries.cell -width 10 -textvariable $this-cell

	pack $w.f.labels.location $w.f.labels.value \
		$w.f.labels.node $w.f.labels.edge \
		$w.f.labels.face $w.f.labels.cell \
		-side top -anchor w

	pack $w.f.entries.loc.locx $w.f.entries.loc.locy $w.f.entries.loc.locz \
		-side left -anchor n -expand yes -fill x

	pack $w.f.entries.loc -side top -expand yes -fill x
	pack $w.f.entries.value \
		$w.f.entries.node $w.f.entries.edge \
		$w.f.entries.face $w.f.entries.cell \
		-side top -anchor w

	pack $w.f.labels $w.f.entries -side left

	bind $w.f.entries.loc.locx <KeyPress-Return> "$this move_location"
	bind $w.f.entries.loc.locy <KeyPress-Return> "$this move_location"
	bind $w.f.entries.loc.locz <KeyPress-Return> "$this move_location"
	bind $w.f.entries.node <KeyPress-Return> "$this move_node"
	bind $w.f.entries.edge <KeyPress-Return> "$this move_edge"
	bind $w.f.entries.face <KeyPress-Return> "$this move_face"
	bind $w.f.entries.cell <KeyPress-Return> "$this move_cell"

	frame $w.controls
	button $w.controls.reset -text "Reset" -command "$this move_center"
	button $w.controls.close -text "Close" -command "destroy $w"
	pack $w.controls.reset $w.controls.close -side left -expand yes -fill x

	pack $w.f $w.controls -side top -expand yes -fill both -padx 5 -pady 5
    }
}




