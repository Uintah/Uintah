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

itcl_class SCIRun_Fields_ClipField {
    inherit Module
    constructor {config} {
        set name ClipField

	global $this-runmode

        set_defaults
    }

    method set_defaults {} {
	set $this-runmode 0
    }

    method replace {} {
	set $this-runmode 1
	$this-c needexecute
    }

    method intersect {} {
	set $this-runmode 2
	$this-c needexecute
    }

    method union {} {
	set $this-runmode 3
	$this-c needexecute
    }

    method invert {} {
	set $this-runmode 4
	$this-c needexecute
    }

    method remove {} {
	set $this-runmode 5
	$this-c needexecute
    }

    method undo {} {
	set $this-runmode 6
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.row1

	pack $w.row1 -side top -e y -f both -padx 5 -pady 5
	
	button $w.row1.replace -text "Replace" -command "$this replace"
	button $w.row1.intersect -text "Intersect" -command "$this intersect"
	button $w.row1.union -text "Union" -command "$this union"
	button $w.row1.invert -text "Invert" -command "$this invert"
	button $w.row1.remove -text "Remove" -command "$this remove"
	button $w.row1.undo -text "Undo" -command "$this undo"
	pack $w.row1.replace \
	     $w.row1.union \
	     $w.row1.remove \
	     $w.row1.intersect \
	     $w.row1.invert \
	     $w.row1.undo \
	     -side top -e n -f both
    }
}


