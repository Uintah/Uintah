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

itcl_class SCIRun_Fields_DirectInterpolate {
    inherit Module
    constructor {config} {
        set name DirectInterpolate

	global $this-use_interp
	global $this-use_closest
	global $this-closeness_distance

        set_defaults
    }

    method set_defaults {} {
	set $this-use_interp 1
	set $this-use_closest 1
	set $this-closeness_distance 1.0e15
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	checkbutton $w.interp -text "Enable Interpolation" \
		-variable $this-use_interp -command "$this-c needexecute"

	checkbutton $w.closest -text "Use Closest Element If Not Interpable" \
		-variable $this-use_closest -command "$this-c needexecute"

	frame $w.chelper
	label $w.chelper.label -text "Maximum Closeness"
	entry $w.chelper.entry -textvariable $this-closeness_distance

	pack $w.chelper.label $w.chelper.entry -side left -anchor n

	pack $w.interp $w.closest $w.chelper -side top -anchor w -padx 10
	
    }
}


