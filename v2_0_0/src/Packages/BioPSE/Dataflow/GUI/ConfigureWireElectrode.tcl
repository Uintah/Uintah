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
 #  ConfigureWireElectrode.tcl: Set theta and phi for the dipole
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

catch {rename BioPSE_Forward_ConfigureWireElectrode ""}

itcl_class BioPSE_Forward_ConfigureWireElectrode {
    inherit Module
    constructor {config} {
        set name ConfigureWireElectrode
        set_defaults
    }
    method set_defaults {} {
	global $this-voltage
	set $this-voltage 5
	global $this-radius
	set $this-radius 0.1
	global $this-nu
	set $this-nu 5
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
	global $this-voltage
	make_entry $w.f.v "Voltage:" $this-voltage \
		"$this-c needexecute"
	global $this-radius
	make_entry $w.f.r "Wire radius:" $this-radius \
		"$this-c needexecute"
	global $this-nu
	make_entry $w.f.nu "Circular segments:" $this-nu \
		"$this-c needexecute"
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.v $w.f.r $w.f.nu $w.f.b -side top
        pack $w.f -side top -fill x -expand yes
    }
}
