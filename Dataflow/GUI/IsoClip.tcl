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

itcl_class SCIRun_Fields_IsoClip {
    inherit Module
    constructor {config} {
        set name IsoClip
        set_defaults
    }

    method set_defaults {} {
        global $this-isoval
	global $this-lte
	set $this-isoval 0
	set $this-lte 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

	entry $w.entry -textvariable $this-isoval
	radiobutton $w.lte -text "Less Than" -value 1 -variable $this-lte
	radiobutton $w.gte -text "Greater Than" -value 0 -variable $this-lte
	button $w.execute -text "Execute" -command "$this-c needexecute"
	pack $w.entry $w.lte $w.gte -side top -anchor w
	pack $w.execute -side bottom -fill x
    }
}
