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

#  ShowWidgets.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Jan. 1995
#  Copyright (C) 1995 SCI Group

itcl_class SCIRun_Visualization_ShowWidgets {
    inherit Module
    constructor {config} {
	set name ShowWidgets
	set_defaults
    }
    method set_defaults {} {
	global $this-widget_scale
	set $this-widget_scale 0.01
	global $this-widget_type
	set $this-widget_type 6

	$this-c select
	$this-c needexecute
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"

	scale $w.f.slide -label Scale -from 0.001 -to 0.05 -length 5c \
		-showvalue true \
		-orient horizontal -resolution 0.001 \
		-digits 8 -variable $this-widget_scale -command "$this-c scale"
	pack $w.f.slide -in $w.f -side top -padx 2 -pady 2 -anchor w

	make_labeled_radio $w.f.wids "Widgets:" "$this-c select;$n" \
		top $this-widget_type \
		{{PointWidget 0} {ArrowWidget 1} \
		{CriticalPointWidget 2} \
		{CrossHairWidget 3} {GaugeWidget 4} \
		{RingWidget 5} {FrameWidget 6} \
		{ScaledFrameWidget 7} {BoxWidget 8} \
		{ScaledBoxWidget 9} {ViewWidget 10} \
		{LightWidget 11} {PathWidget 12}}
	pack $w.f.wids

	button $w.f.nextmode -text "NextMode" -command "$this-c nextmode"
	pack $w.f.nextmode -fill x -pady 2
	button $w.f.ui -text "UI" -command "$this-c ui"
	pack $w.f.ui -fill x -pady 2
    }
}
