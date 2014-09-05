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

itcl_class SCIRun_FieldsCreate_SampleStructHex {
    inherit Module
    constructor {config} {
        set name SampleStructured
        set_defaults
    }

    method set_defaults {} {
	global $this-sizex
	global $this-sizey
	global $this-sizez
	global $this-padpercent
	global $this-data-at
	set $this-sizex 16
	set $this-sizey 16
	set $this-sizez 16
	set $this-padpercent 0
	set $this-data-at Nodes
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
	frame $w.row31
	frame $w.row4
	frame $w.which -relief groove -borderwidth 2

	pack $w.row1 $w.row2 $w.row3 $w.row31 $w.which $w.row4 -side top \
		-e y -f both -padx 5 -pady 5
	
	label $w.row1.xsize_label -text "X Size   "
	entry $w.row1.xsize -textvariable $this-sizex
	label $w.row2.ysize_label -text "Y Size   "
	entry $w.row2.ysize -textvariable $this-sizey
	label $w.row3.zsize_label -text "Z Size   "
	entry $w.row3.zsize -textvariable $this-sizez

	label $w.row31.zsize_label -text "Pad Percentage"
	entry $w.row31.zsize -textvariable $this-padpercent

	pack $w.row1.xsize_label $w.row1.xsize -side left
	pack $w.row2.ysize_label $w.row2.ysize -side left
	pack $w.row3.zsize_label $w.row3.zsize -side left
	pack $w.row31.zsize_label $w.row31.zsize -side left

	button $w.row4.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row4.execute -side top -e n -f both

	label $w.which.l -text "Data at Location"
	radiobutton $w.which.node -text "Nodes" \
		-variable $this-data-at -value Nodes
	radiobutton $w.which.edge -text "Edges" \
		-variable $this-data-at -value Edges
	radiobutton $w.which.face -text "Faces" \
		-variable $this-data-at -value Faces
	radiobutton $w.which.cell -text "Cells" \
		-variable $this-data-at -value Cells
	radiobutton $w.which.none -text "None" \
		-variable $this-data-at -value None
	pack $w.which.l -side top
	pack $w.which.node $w.which.edge $w.which.face $w.which.cell \
	    $w.which.none -anchor nw
    }
}


