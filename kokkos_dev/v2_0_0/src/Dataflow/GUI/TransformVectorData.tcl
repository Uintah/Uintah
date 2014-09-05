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

itcl_class SCIRun_FieldsData_TransformVectorData {
    inherit Module
    constructor {config} {
        set name TransformVectorData
        set_defaults
    }
    method set_defaults {} {
        global $this-functionx
        global $this-functiony
        global $this-functionz
	global $this-pre_normalize
	global $this-post_normalize
	set $this-functionx "x"
	set $this-functiony "y"
	set $this-functionz "z"
	set $this-pre_normalize 0
	set $this-post_normalize 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        #wm minsize $w 170 60
        frame $w.f

	global $this-functionx
	global $this-functiony
	global $this-functionz

	set c "$this-c needexecute"

	checkbutton $w.f.pre -text "Prenormalize" \
	    -variable $this-pre_normalize -anchor w -just left

	checkbutton $w.f.post -text "Postnormalize" \
	    -variable $this-post_normalize -anchor w -just left

	frame $w.f.x
	label $w.f.x.r -text "X Function         " -anchor w -just left;
	entry $w.f.x.e -textvariable $this-functionx
	bind $w.f.x.e <Return> $c
	pack $w.f.x.r $w.f.x.e -side left -fill x -expand 1

	frame $w.f.y
	label $w.f.y.r -text "Y Function         " -anchor w -just left;
	entry $w.f.y.e -textvariable $this-functiony
	bind $w.f.y.e <Return> $c
	pack $w.f.y.r $w.f.y.e -side left -fill x -expand 1

	frame $w.f.z
	label $w.f.z.r -text "Z Function         " -anchor w -just left;
	entry $w.f.z.e -textvariable $this-functionz
	bind $w.f.z.e <Return> $c
	pack $w.f.z.r $w.f.z.e -side left -fill x -expand 1

	pack $w.f.x $w.f.y $w.f.z $w.f.pre $w.f.post \
	    -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
