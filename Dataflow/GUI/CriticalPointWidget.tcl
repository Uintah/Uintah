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

#  CriticalPointWidget.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#  Copyright (C) 1995 SCI Group


catch {rename CriticalPointWidget ""}

itcl_class CriticalPointWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name CriticalPointWidget
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 100 100

	Dialbox $w.dialbox "CriticalPointWidget - Translate"
	$w.dialbox unbounded_dial 0 "Translate X" 0.0 1.0 \
		"$this-c translate x" "$this-c dialdone"
	$w.dialbox unbounded_dial 2 "Translate Y" 0.0 1.0 \
		"$this-c translate y" "$this-c dialdone"
	$w.dialbox unbounded_dial 4 "Translate Z" 0.0 1.0 \
		"$this-c translate z" "$this-c dialdone"

	frame $w.f
	Base_ui $w.f
	button $w.f.dials -text "Dialbox" -command "$w.dialbox connect"
	pack $w.f.dials -pady 5
	pack $w.f
    }
}
