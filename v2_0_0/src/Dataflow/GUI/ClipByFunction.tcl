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

itcl_class SCIRun_FieldsCreate_ClipByFunction {
    inherit Module
    constructor {config} {
        set name ClipByFunction
        set_defaults
    }

    method set_defaults {} {
        global $this-clipfunction
	global $this-clipmode
	set $this-clipfunction "x < 0"
	set $this-clipmode "cell"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

	set c "$this-c needexecute"

	frame $w.location -relief groove -borderwidth 2
	label $w.location.label -text "Clip Location"
	radiobutton $w.location.cell -text "Cell Centers" \
	    -variable $this-clipmode -value cell -command $c
	radiobutton $w.location.nodeone -text "One Node" \
	    -variable $this-clipmode -value onenode -command $c
	radiobutton $w.location.nodeall -text "All Nodes" \
	    -variable $this-clipmode -value allnodes -command $c

	pack $w.location.label -side top -expand yes -fill both
	pack $w.location.cell $w.location.nodeone $w.location.nodeall \
	    -side top -anchor w

	frame $w.function -borderwidth 2
	label $w.function.l -text "F(x, y, z, v)"
	entry $w.function.e -width 20 -textvariable $this-clipfunction
	bind $w.function.e <Return> $c
	pack $w.function.l $w.function.e -side left

	pack $w.location $w.function -side top -fill x -expand 1 \
	    -padx 5 -pady 5
    }
}
