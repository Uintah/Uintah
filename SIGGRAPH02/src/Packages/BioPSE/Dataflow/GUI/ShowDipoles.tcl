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

##
 #  ShowDipoles.tcl: Set theta and phi for the dipole
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename BioPSE_Visualization_ShowDipoles ""}

itcl_class BioPSE_Visualization_ShowDipoles {
    inherit Module
    constructor {config} {
        set name ShowDipoles
        set_defaults
    }
    method set_defaults {} {
	global $this-widgetSizeGui_
	global $this-scaleModeGui_
	global $this-showLastVecGui_
	global $this-showLinesGui_
	set $this-widgetSizeGui_ 1
	set $this-scaleModeGui_ normalize
	set $this-showLastVecGui_ 0
	set $this-showLinesGui_ 1
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
	global $v
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
        set w .ui$[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 30
        frame $w.f
	global $this-widgetSizeGui_
	make_entry $w.f.s "Widget Size:" $this-widgetSizeGui_ "$this-c needexecute"
	frame $w.f.r -relief sunken -bd 2
	global $this-scaleModeGui_
	radiobutton $w.f.r.fixed -text "Fixed Size" -value "fixed" -variable $this-scaleModeGui_
	radiobutton $w.f.r.normalize -text "Normalize Largest" -value "normalize" -variable $this-scaleModeGui_
	radiobutton $w.f.r.scale -text "Scale Size" -value "scale" -variable $this-scaleModeGui_
	pack $w.f.r.fixed $w.f.r.normalize $w.f.r.scale -side top -fill both -expand yes
	global $this-showLastVecGui_
	checkbutton $w.f.v -text "Show Last As Vector" -variable $this-showLastVecGui_
	global $this-showLinesGui_
	checkbutton $w.f.l -text "Show Lines" -variable $this-showLinesGui_

	pack $w.f.s $w.f.r $w.f.v $w.f.l -side top -fill x -expand yes
        pack $w.f -side top -fill x -expand yes
    }
}
