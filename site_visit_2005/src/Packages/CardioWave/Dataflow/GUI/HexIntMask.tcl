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

catch {rename CardioWave_CreateModel_HexIntMask ""}

itcl_class CardioWave_CreateModel_HexIntMask {
    inherit Module

    constructor {config} {
        set name HexIntMask

	global $this-exclude
	global $this-delete-nodes
	global $this-levels
        set_defaults
    }

    method set_defaults {} {
	set $this-exclude ""
	set $this-delete-nodes 0
	set $this-levels 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.chelper
	label $w.chelper.label -text "Exclude List"
	entry $w.chelper.entry -textvariable $this-exclude
	
	pack $w.chelper.label $w.chelper.entry -side left -anchor n

	scale $w.levels -label "Hull Levels:" -variable $this-levels \
	    -from 0 -to 20 -showvalue true \-orient horizontal
	pack $w.levels -side top

	pack $w.chelper -side top -anchor w -padx 10
	checkbutton $w.delete -text "Delete Unattached Nodes" -variable $this-delete-nodes
	pack $w.delete -side top
    }
}


