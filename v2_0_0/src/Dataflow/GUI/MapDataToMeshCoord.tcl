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

itcl_class SCIRun_FieldsGeometry_MapDataToMeshCoord {
    inherit Module
    constructor {config} {
        set name MapDataToMeshCoord
        set_defaults
    }
    method set_defaults {} {
        global $this-coord
	set $this-coord 2
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 170 20
        frame $w.f

	global $this-coord
	frame $w.l
	label $w.l.label -text "Coordinate to replace with data: "
	pack $w.l.label -side left
	frame $w.b
	radiobutton $w.b.x -text "X    " -variable $this-coord -value 0
	radiobutton $w.b.y -text "Y    " -variable $this-coord -value 1
	radiobutton $w.b.z -text "Z    " -variable $this-coord -value 2
	radiobutton $w.b.n -text "Push surface nodes by (normal x data)" -variable $this-coord -value 3
	pack $w.b.x $w.b.y $w.b.z $w.b.n -side left -expand 1 -fill x
	pack $w.l $w.b -side top -fill both -expand 1
    }
}
