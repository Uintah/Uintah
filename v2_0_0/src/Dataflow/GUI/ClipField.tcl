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

itcl_class SCIRun_FieldsCreate_ClipField {
    inherit Module
    constructor {config} {
        set name ClipField

	global $this-clip-location  # Where to clip
	global $this-clipmode       # Which clip mode to use.
	global $this-autoexecute    # Execute on widget button up?
	global $this-autoinvert     # Invert again when executing?
	global $this-execmode       # Which of three executes to use.

        set_defaults
    }

    method set_defaults {} {
	set $this-clip-location cell
	set $this-clipmode replace
	set $this-autoexecute 0
	set $this-autoinvert 0
	set $this-execmode 0
    }

    method execrunmode {} {
	set $this-execmode execute
	$this-c needexecute
    }
    method invert {} {
	set $this-execmode invert
	$this-c needexecute
    }

    method undo {} {
	set $this-execmode undo
	$this-c needexecute
    }

    method reset {} {
	set $this-execmode reset
	$this-c needexecute
    }

    method locationclip {} {
	set $this-execmode location
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.location -relief groove -borderwidth 2
	frame $w.execmode -relief groove -borderwidth 2
	frame $w.whenexecute
	frame $w.executes -relief groove -borderwidth 2

	label $w.location.label -text "Clip Location"
	radiobutton $w.location.cell -text "Cell Centers" -variable $this-clip-location -value cell -command "$this locationclip"
	radiobutton $w.location.nodeone -text "One Node" -variable $this-clip-location -value nodeone -command "$this locationclip"
	radiobutton $w.location.nodeall -text "All Nodes" -variable $this-clip-location -value nodeall -command "$this locationclip"

	pack $w.location.label -side top -expand yes -fill both
	pack $w.location.cell $w.location.nodeone $w.location.nodeall -side top -anchor w

	label $w.execmode.label -text "Execute Action"
	radiobutton $w.execmode.replace -text "Replace" -variable $this-clipmode -value replace
	radiobutton $w.execmode.union -text "Union" -variable $this-clipmode -value union
	radiobutton $w.execmode.intersect -text "Intersect" -variable $this-clipmode -value intersect
	radiobutton $w.execmode.remove -text "Remove" -variable $this-clipmode -value remove

	pack $w.execmode.label -side top -fill both
	pack $w.execmode.replace $w.execmode.union $w.execmode.intersect $w.execmode.remove -side top -anchor w 

	checkbutton $w.whenexecute.check -text "Execute automatically" -variable $this-autoexecute

	checkbutton $w.whenexecute.icheck -text "Invert automatically" -variable $this-autoinvert -command "$this locationclip"
	
	pack $w.whenexecute.check $w.whenexecute.icheck -side top -anchor w -padx 10

	button $w.executes.execute -text "Execute" -command "$this execrunmode"
	button $w.executes.invert -text "Invert" -command "$this invert"
	button $w.executes.undo -text "Undo" -command "$this undo"
	button $w.executes.reset -text "Reset" -command "$this reset"
	pack $w.executes.execute $w.executes.invert $w.executes.undo $w.executes.reset -side left -e y -f both -padx 5 -pady 5

	pack $w.location $w.execmode $w.whenexecute $w.executes -side top -e y -f both -padx 5 -pady 5
	
    }
}


