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
 #  InsertVoltageSource.tcl: Set theta and phi for the dipole
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

catch {rename BioPSE_Forward_InsertVoltageSource ""}

itcl_class BioPSE_Forward_InsertVoltageSource {
    inherit Module
    constructor {config} {
        set name InsertVoltageSource
        set_defaults
    }
    method set_defaults {} {
	global $this-outside
	set $this-outside 1
	global $this-groundfirst
	set $this-groundfirst 0
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
	global $this-outside
	checkbutton $w.f.o -text "Interpolate outside mesh" \
		-variable $this-outside
	global $this-groundfirst
	checkbutton $w.f.g -text "Ground first node of second field" \
		-variable $this-groundfirst
	
	pack $w.f.o $w.f.g -side top -fill x -expand yes -padx 5 -pady 5
        pack $w.f -side top -fill x -expand yes
    }
}
