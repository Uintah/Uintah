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
 #  ConfigureElectrode.tcl: Set theta and phi for the dipole
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

catch {rename BioPSE_Forward_ConfigureElectrode ""}

itcl_class BioPSE_Forward_ConfigureElectrode {
    inherit Module
    constructor {config} {
        set name ConfigureElectrode
        set_defaults
    }
    method set_defaults {} {
	global $this-active
	set $this-active "front"
	global $this-voltage
	set $this-voltage 5
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
	label $w.f.l -text "Active Side of Electrode:"
	global $this-active
	radiobutton $w.f.front -text "Front" \
	    -variable $this-active -value "front"
	radiobutton $w.f.back -text "Back" \
	    -variable $this-active -value "back"
	radiobutton $w.f.both -text "Both Sides" \
	    -variable $this-active -value "both"
	global $this-voltage
	make_entry $w.f.v "Electrode Voltage:" $this-voltage \
		"$this-c needexecute"
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.l $w.f.front $w.f.back $w.f.both $w.f.v $w.f.b -side top
        pack $w.f -side top -fill x -expand yes
    }
}
