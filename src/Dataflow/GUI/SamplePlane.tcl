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

itcl_class SCIRun_Fields_SamplePlane {
    inherit Module
    constructor {config} {
        set name SamplePlane
        set_defaults
    }

    method set_defaults {} {
	global $this-sizex
	global $this-sizey
	global $this-axis
	global $this-padpercent
	global $this-data-at
	global $this-update_type
	global $this-pos

	set $this-sizex 20
	set $this-sizey 20
	set $this-axis 0
	set $this-padpercent 0
	set $this-data-at Nodes
	set $this-update_type "on release"
	set $this-pos 0

    }

    method update-type { w } {
	global $this-update_type
	set $this-update_type [$w get]
    }

    method position_release { } {
	global $this-update_type

	set type [set $this-update_type]
	if { $type == "on release" } {
	    eval "$this-c needexecute"
	}
    }

    method set_position {v} {
	global $this-update_type

	set type [set $this-update_type]
	if { $type == "Auto" } {
	    eval "$this-c needexecute"
	}
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
	frame $w.row21
	frame $w.row3
	frame $w.row4
	frame $w.which -relief groove -borderwidth 2
	frame $w.row5

	pack $w.row1 $w.row2 $w.row21 $w.row3 $w.row4 $w.which $w.row5 \
	    -side top -e y -f both -padx 5 -pady 5
	
	label $w.row1.xsize_label -text "Width    "
	entry $w.row1.xsize -textvariable $this-sizex
	label $w.row2.ysize_label -text "Height   "
	entry $w.row2.ysize -textvariable $this-sizey

	pack $w.row1.xsize_label $w.row1.xsize -side left
	pack $w.row2.ysize_label $w.row2.ysize -side left

	label $w.row21.zsize_label -text "Pad Percentage"
	entry $w.row21.zsize -textvariable $this-padpercent
	pack $w.row21.zsize_label $w.row21.zsize -side left

	label $w.row3.label -text "Axis: "
	radiobutton $w.row3.x -text "X    " -variable $this-axis -value 0
	radiobutton $w.row3.y -text "Y    " -variable $this-axis -value 1
	radiobutton $w.row3.z -text "Z" -variable $this-axis -value 2
	pack $w.row3.label $w.row3.x $w.row3.y $w.row3.z -side left

	label $w.row4.label -text "Position: "
	scale $w.row4.scale -from -1.0 -to 1.0 -resolution .01 -width 10 -orient horizontal -command "$this set_position" -variable $this-pos

	iwidgets::optionmenu $w.row4.update -labeltext "Update:" \
	    -labelpos w -command "$this update-type $w.row4.update"
	$w.row4.update insert end "on release" Manual Auto
	$w.row4.update select [set $this-update_type]

	bind $w.row4.scale <ButtonRelease> "$this position_release"

	
	pack $w.row4.label $w.row4.scale $w.row4.update -side left

	label $w.which.l -text "Data At Location"
	radiobutton $w.which.node -text "Nodes" \
		-variable $this-data-at -value Nodes
	radiobutton $w.which.edge -text "Edges" \
		-variable $this-data-at -value Edges
	radiobutton $w.which.face -text "Faces" \
		-variable $this-data-at -value Faces
	radiobutton $w.which.none -text "None" \
		-variable $this-data-at -value None
	pack $w.which.l -side top
	pack $w.which.node $w.which.edge $w.which.face \
	    $w.which.none -anchor nw

	button $w.row5.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row5.execute -side top -e n -f both
    }
}


