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

itcl_class SCIRun_FieldsData_ManageFieldData {
    inherit Module
    constructor {config} {
        set name ManageFieldData
        set_defaults
    }
    method set_defaults {} {
        global $this-preserve-scalar-type
	set $this-preserve-scalar-type 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

	checkbutton $w.preserve -text "Preserve Scalar Field Type" \
	    -variable $this-preserve-scalar-type -command "$this-c needexecute"

	pack $w.preserve

	makeSciButtonPanel $w $w $this
	moveToCursor $w
     }
}
