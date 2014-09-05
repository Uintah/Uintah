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
 #  IntegrateCurrent.tcl: Set theta and phi for the dipole
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

catch {rename BioPSE_Forward_IntegrateCurrent ""}

itcl_class BioPSE_Forward_IntegrateCurrent {
    inherit Module
    constructor {config} {
        set name IntegrateCurrent
        set_defaults
    }
    method set_defaults {} {
	global $this-current
	set $this-current 0
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
	global $this-current
	frame $w.f.c
	label $w.f.c.l -text "Computed current: "
	label $w.f.c.c -textvariable $this-current
	label $w.f.c.a -text "amps"
	pack $w.f.c.l $w.f.c.c $w.f.c.a -side left -fill x -expand yes
	pack $w.f.c -side top -fill x -expand yes
        pack $w.f -side top -fill x -expand yes
    }
}
