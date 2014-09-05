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

itcl_class SCIRun_Visualization_ShowColorMap {
    inherit Module
    constructor {config} {
        set name ShowColorMap

	global $this-length       # One of three lengths to use.
	global $this-side         # Which side to put the map on.
	global $this-numlabels    # How many labels to use on the map.

        set_defaults
    }

    method set_defaults {} {
	set $this-length half2
	set $this-side left
	set $this-numlabels 5
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.side -relief groove -borderwidth 2
	frame $w.length -relief groove -borderwidth 2
	frame $w.nlabs -borderwidth 2


	label $w.side.label -text "Display Side"
	radiobutton $w.side.left -text "Left" -variable $this-side -value left -command "$this-c needexecute"
	radiobutton $w.side.bottom -text "Bottom" -variable $this-side -value bottom -command "$this-c needexecute"

	pack $w.side.label -side top -expand yes -fill both
	pack $w.side.left $w.side.bottom -side top -anchor w


	label $w.length.label -text "Display Length"
	radiobutton $w.length.full -text "Full" -variable $this-length -value full -command "$this-c needexecute"
	radiobutton $w.length.half1 -text "First Half" -variable $this-length -value half1 -command "$this-c needexecute"
	radiobutton $w.length.half2 -text "Second Half" -variable $this-length -value half2 -command "$this-c needexecute"

	pack $w.length.label -side top -expand yes -fill both
	pack $w.length.full $w.length.half1 $w.length.half2 -side top -anchor w


	label $w.nlabs.label -text "Labels"
	entry $w.nlabs.entry -width 5 -textvariable $this-numlabels
	bind $w.nlabs.entry <KeyPress-Return> "$this-c needexecute"

	pack $w.nlabs.label $w.nlabs.entry -side left -anchor n

	pack $w.side $w.length $w.nlabs -side top -e y -f both -padx 5 -pady 5
    }
}


