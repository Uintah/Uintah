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

itcl_class SCIRun_Fields_BuildInterpolant {
    inherit Module
    constructor {config} {
        set name BuildInterpolant

	global $this-use_closest_outside
	global $this-interp_nearest

        set_defaults
    }

    method set_defaults {} {
	set $this-use_closest_outside 1
	set $this-interp_nearest 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.options

	checkbutton $w.options.outside -text "Use Closest Element When Outside Field" -variable $this-use_closest_outside -command "$this-c needexecute"

	checkbutton $w.options.nearest -text "Interp Nearest" -variable $this-interp_nearest -command "$this-c needexecute"

	pack $w.options.outside $w.options.nearest -side top -anchor w -padx 10

	pack $w.options -side top -e y -f both -padx 5 -pady 5
	
    }
}


