#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#


itcl_class SCIRun_NewField_GenerateSinglePointProbeFromField {
    inherit Module

    constructor {config} {
        set name GenerateSinglePointProbeFromField	
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
            return
        }

        toplevel $w

	build_ui $w

	makeSciButtonPanel $w $w $this -no_execute "\"Reset\" \"$this move_center\" \"\""
	moveToCursor $w
    }

    method build_ui { w } {
	global $this- main_frame
	set $this-main_frame $w
	
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


	scale $w.f.slide -orient horizontal -label "GenerateSinglePointProbeFromField Size" -from 0 -to 100 -showvalue true \
	     -variable $this-probe_scale -resolution 0.25 -tickinterval 25
	set $w.f.slide $this-probe_scale

	bind $w.f.slide <ButtonRelease> "$this-c needexecute"
	bind $w.f.slide <B1-Motion> "$this-c needexecute"

	pack $w.f.slide $w.f.g -side bottom -expand yes -fill x

	pack $w.f -side top -expand yes -fill both -padx 5 -pady 5

	bind $w.f.g.entries.loc.locx <KeyPress-Return> "$this move_location"
	bind $w.f.g.entries.loc.locy <KeyPress-Return> "$this move_location"
	bind $w.f.g.entries.loc.locz <KeyPress-Return> "$this move_location"
	bind $w.f.g.entries.node <KeyPress-Return> "$this move_node"
	bind $w.f.g.entries.edge <KeyPress-Return> "$this move_edge"
	bind $w.f.g.entries.face <KeyPress-Return> "$this move_face"
	bind $w.f.g.entries.cell <KeyPress-Return> "$this move_cell"
    }
}




