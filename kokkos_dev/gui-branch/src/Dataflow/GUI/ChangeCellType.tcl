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
 #  ChangeCellType.tcl: The ChangeCellType UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename SCIRun_EEG_ChangeCellType ""}

itcl_class SCIRun_EEG_ChangeCellType {
    inherit ModuleGui
    constructor {config} {
        set name ChangeCellType
        set_defaults
    }
    method set_defaults {} {
        global $this-npts
        set $this-scalarAsCondTCL 1
	set $this-removeAirTCL 1
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 300 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-scalarAsCondTCL
	checkbutton $w.f.s -text "Scalar As Conductivity Index" -variable $this-scalarAsCondTCL
	global $this-removeAirTCL
	checkbutton $w.f.r -text "Remove Air Elements" -variable $this-removeAirTCL
	pack $w.f.s $w.f.r -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
