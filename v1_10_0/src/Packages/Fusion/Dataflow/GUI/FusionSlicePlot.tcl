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

# GUI for FusionSlicePlot module
# by Allen R. Sanderson
# March 2002

# This GUI interface consists of a widget that allows for scaling of 
# a height field in a surface.

package require Iwidgets 3.0   

itcl_class Fusion_Fields_FusionSlicePlot {
    inherit Module
    constructor {config} {
        set name FusionSlicePlot
        set_defaults
    }

    method set_defaults {} {

	global $this-scale
	global $this-update_type

	set $this-scale 1.00
	set $this-update_type "on release"
    }

    method ui {} {

	set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
	}

	toplevel $w

	expscale $w.slide -label Scale \
	    -orient horizontal -variable $this-scale \
	    -command "$this dragSlider"

	bind $w.slide.scale <ButtonRelease> "$this releaseSlider"

	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x

	#  Options
	iwidgets::labeledframe $w.f.opt -labelpos nw -labeltext "Options"
	set opt [$w.f.opt childsite]
	
	iwidgets::optionmenu $opt.update -labeltext "Update:" \
	    -labelpos w -command "$this update-type $opt.update"
	$opt.update insert end "on release" Manual Auto
	$opt.update select [set $this-update_type]

	global $this-update
	set $this-update $opt.update

	pack $opt.update -side top -anchor w
	pack $w.f.opt -side top -fill x -expand 1

	frame $w.misc
	button $w.misc.execute -text "Execute" -command "$this-c needexecute"
	button $w.misc.dismiss -text Dismiss -command "destroy $w"
	pack $w.misc.execute $w.misc.dismiss -side left -padx 25

	pack $w.misc -side top -padx 10 -pady 5	    
    }

    method dragSlider {someUknownVar} {

	if { [set $this-update_type] == "Auto" } {
	    eval "$this-c needexecute"
	}
    }

    method releaseSlider {} {

	if { [set $this-update_type] == "on release" } {
	    eval "$this-c needexecute"
	}
    }

    method update-type { w } {
	global $w
	global $this-update_type

	set $this-update_type [$w get]
    }
}