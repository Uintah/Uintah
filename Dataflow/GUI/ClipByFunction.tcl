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

itcl_class SCIRun_Fields_ClipByFunction {
    inherit Module
    constructor {config} {
        set name ClipByFunction
        set_defaults
    }
    method set_defaults {} {
        global $this-clipfunction
	global $this-clipmode
	set $this-clipfunction "x < 0"
	set $this-clipmode "allnode"
    }

    method functioneval2 {x y z v function} {
	if {![catch {expr $function} result]} {
	    return $result
	}
	return 0
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

	set c "$this-c needexecute"

	entry $w.e -textvariable $this-clipfunction
	bind $w.e <Return> $c
	pack $w.e -side left -fill x -expand 1
    }
}
