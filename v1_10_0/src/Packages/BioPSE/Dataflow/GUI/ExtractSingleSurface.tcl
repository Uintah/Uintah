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
 #  ExtractSingleSurface.tcl: Extract a single component from a separating 
 #                            surface
 #
 #  Written by:
 #   David Weinstein
 #   School of Computing
 #   University of Utah
 #   March 2003
 #
 #  Copyright (C) 2003 SCI Institute
 # 
 ##

catch {rename BioPSE_Modeling_ExtractSingleSurface ""}

itcl_class BioPSE_Modeling_ExtractSingleSurface {
    inherit Module
    constructor {config} {
        set name ExtractSingleSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-surfid
	global $this-data
	set $this-surfid "0"
	set $this-data "material"
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-surfid
	frame $w.f.surf
	label $w.f.surf.l -text "SurfId: "
	entry $w.f.surf.e -relief sunken -width 10 -textvariable $this-surfid
	bind $w.f.surf.e <Return> $n
	pack $w.f.surf.l $w.f.surf.e -side left
	pack $w.f.surf -side left -fill x
	frame $w.data
	label $w.data.label -text "Data values based on:"
	radiobutton $w.data.material -text "Material" \
		-variable $this-data -value material
	radiobutton $w.data.cindex -text "Component index" \
		-variable $this-data -value cindex
	radiobutton $w.data.size -text "Size (voxels)" \
		-variable $this-data -value size
	pack $w.data.label $w.data.material $w.data.cindex $w.data.size \
	    -side top -anchor w
        pack $w.f $w.data -side top -expand yes
    }
}
