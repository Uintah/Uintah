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

itcl_class SCIRun_FieldsGeometry_CastTVtoMLV {
    inherit Module
    constructor {config} {
        set name CastTVtoMLV
	global $this-nx
	global $this-ny
	global $this-nz
        set_defaults
    }

    method set_defaults {} {
	set $this-nx 8
	set $this-ny 8
	set $this-nz 8
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w
	
	frame $w.x
	frame $w.y
	frame $w.z

	pack $w.x $w.y $w.z -side top -e y -f both -padx 5 -pady 5
	
	label $w.x.label -text "nx: "
	entry $w.x.entry -textvariable $this-nx
	bind $w.x.entry <Return> "$this-c needexecute"
	pack $w.x.label $w.x.entry -side left

	label $w.y.label -text "ny: "
	entry $w.y.entry -textvariable $this-ny
	bind $w.y.entry <Return> "$this-c needexecute"
	pack $w.y.label $w.y.entry -side left

	label $w.z.label -text "nz: "
	entry $w.z.entry -textvariable $this-nz
	bind $w.z.entry <Return> "$this-c needexecute"
	pack $w.z.label $w.z.entry -side left
    }
}


