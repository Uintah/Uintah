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

itcl_class SCIRun_Math_Submatrix {
    inherit Module
    constructor {config} {
        set name Submatrix
        set_defaults
    }

    method set_defaults {} {
	global $this-minrow
	global $this-maxrow
	global $this-mincol
	global $this-maxcol
	global $this-nrow
	global $this-ncol

	set $this-minrow "--"
	set $this-maxrow "--"
	set $this-mincol "--"
	set $this-maxcol "--"
	set $this-nrow "??"
	set $this-ncol "??"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.info
	frame $w.row
	frame $w.col

	global $this-nrow $this-ncol
	label $w.info.label -text "Input Matrix:   nrows ="
	label $w.info.row -textvariable $this-nrow
	label $w.info.x -text "  ncols ="
	label $w.info.col -textvariable $this-ncol

	label $w.row.label -text "Row Range" -width 10 -just left
	entry $w.row.min -width 10 -textvariable $this-minrow
	entry $w.row.max -width 10 -textvariable $this-maxrow

	label $w.col.label -text "Col Range" -width 10 -just left
	entry $w.col.min -width 10 -textvariable $this-mincol
	entry $w.col.max -width 10 -textvariable $this-maxcol

	bind $w.row.min <KeyPress-Return> "$this-c needexecute"
	bind $w.row.max <KeyPress-Return> "$this-c needexecute"
	bind $w.col.min <KeyPress-Return> "$this-c needexecute"
	bind $w.col.max <KeyPress-Return> "$this-c needexecute"

	pack $w.info.label $w.info.row $w.info.x $w.info.col -side left -anchor n -expand yes -fill x

	pack $w.row.label $w.row.min $w.row.max -side left -anchor n -expand yes -fill x

	pack $w.col.label $w.col.min $w.col.max -side left -anchor n -expand yes -fill x
	
	button $w.execute -text "Execute" -command "$this-c needexecute"

	pack $w.info $w.row $w.col $w.execute -side top -expand yes -fill x
    }
}




