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
	global $this-show-value
	global $this-locx
	global $this-locy
	global $this-locz
	global $this-node
	global $this-edge
	global $this-face
	global $this-cell
	global $this-show-node
	global $this-show-edge
	global $this-show-face
	global $this-show-cell
	global $this-moveto
	global $this-probe_scale
	

	set $this-value ""
	set $this-show-value 1
	set $this-locx ""
	set $this-locy ""
	set $this-locz ""
	set $this-node ""
	set $this-edge ""
	set $this-face ""
	set $this-cell ""
	set $this-show-node 1
	set $this-show-edge 1
	set $this-show-face 1
	set $this-show-cell 1
	set $this-moveto ""
	set $this-probe_scale 5.0
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


    method changevalue {} {
	if { [set $this-show-value] } {
	    $this-c needexecute
	} else {
	    set $this-value ""
	}
    }

    method changenode {} {
	if { [set $this-show-node] } {
	    $this-c needexecute
	} else {
	    set $this-node ""
	}
    }

    method changeedge {} {
	if { [set $this-show-edge] } {
	    $this-c needexecute
	} else {
	    set $this-edge ""
	}
    }

    method changeface {} {
	if { [set $this-show-face] } {
	    $this-c needexecute
	} else {
	    set $this-face ""
	}
    }

    method changecell {} {
	if { [set $this-show-cell] } {
	    $this-c needexecute
	} else {
	    set $this-cell ""
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f
	frame $w.f.g
	frame $w.f.g.labels
	frame $w.f.g.entries
	frame $w.f.g.entries.loc


	label $w.f.g.labels.location -text "Location" -just left
	entry $w.f.g.entries.loc.locx -width 10 -textvariable $this-locx
	entry $w.f.g.entries.loc.locy -width 10 -textvariable $this-locy
	entry $w.f.g.entries.loc.locz -width 10 -textvariable $this-locz

	checkbutton $w.f.g.labels.value -text "Value" -just left \
	    -variable $this-show-value -command "$this changevalue"
	entry $w.f.g.entries.value -width 40 -state disabled -textvariable $this-value

	checkbutton  $w.f.g.labels.node -text "Node" -just left \
	    -variable $this-show-node -command "$this changenode"
	entry $w.f.g.entries.node -width 10 -textvariable $this-node

	checkbutton $w.f.g.labels.edge -text "Edge" -just left \
	    -variable $this-show-edge -command "$this changeedge"
	entry $w.f.g.entries.edge -width 10 -textvariable $this-edge

	checkbutton $w.f.g.labels.face -text "Face" -just left \
	    -variable $this-show-face -command "$this changeface"
	entry $w.f.g.entries.face -width 10 -textvariable $this-face

	checkbutton $w.f.g.labels.cell -text "Cell" -just left \
	    -variable $this-show-cell -command "$this changecell"
	entry $w.f.g.entries.cell -width 10 -textvariable $this-cell

     	pack  $w.f.g.labels.location $w.f.g.labels.value \
	        $w.f.g.labels.node $w.f.g.labels.edge \
		$w.f.g.labels.face $w.f.g.labels.cell\
		-side top -anchor w

	pack $w.f.g.entries.loc.locx $w.f.g.entries.loc.locy $w.f.g.entries.loc.locz \
		-side left -anchor n -expand yes -fill x

	pack $w.f.g.entries.loc -side top -expand yes -fill x
	pack $w.f.g.entries.value $w.f.g.entries.node $w.f.g.entries.edge \
		$w.f.g.entries.face $w.f.g.entries.cell \
		-side top -anchor w

	pack $w.f.g.labels $w.f.g.entries -side left


	scale $w.f.slide -orient horizontal -label "Probe Size" -from 0 -to 100 -showvalue true \
	     -variable $this-probe_scale -resolution 0.25 -tickinterval 25
	set $w.f.slide $this-probe_scale

	bind $w.f.slide <ButtonRelease> "$this-c needexecute"
	bind $w.f.slide <B1-Motion> "$this-c needexecute"

	pack $w.f.slide $w.f.g -side bottom -expand yes -fill x

	bind $w.f.g.entries.loc.locx <KeyPress-Return> "$this move_location"
	bind $w.f.g.entries.loc.locy <KeyPress-Return> "$this move_location"
	bind $w.f.g.entries.loc.locz <KeyPress-Return> "$this move_location"
	bind $w.f.g.entries.node <KeyPress-Return> "$this move_node"
	bind $w.f.g.entries.edge <KeyPress-Return> "$this move_edge"
	bind $w.f.g.entries.face <KeyPress-Return> "$this move_face"
	bind $w.f.g.entries.cell <KeyPress-Return> "$this move_cell"

	frame $w.controls
	button $w.controls.reset -text "Reset" -command "$this move_center"
	button $w.controls.close -text "Close" -command "destroy $w"
	pack $w.controls.reset $w.controls.close -side left -expand yes -fill x


	pack $w.f $w.controls -side top -expand yes -fill both -padx 5 -pady 5
    }
}




