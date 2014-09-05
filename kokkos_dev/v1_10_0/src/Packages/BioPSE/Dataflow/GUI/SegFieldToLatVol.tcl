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

itcl_class BioPSE_Modeling_SegFieldToLatVol {
    inherit Module
    constructor {config} {
        set name SegFieldToLatVol
        set_defaults
    }

    method set_defaults {} {
	global $this-lat_vol_data
	set $this-lat_vol_data componentMatl
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	frame $w.f
	label $w.f.l -text "Data values for output field"
	radiobutton $w.f.sv -text "Material" \
	    -variable $this-lat_vol_data -value componentMatl
	radiobutton $w.f.ci -text "Component Index" \
	    -variable $this-lat_vol_data -value componentIdx
	radiobutton $w.f.cs -text "Component Size" \
	    -variable $this-lat_vol_data -value componentSize
	pack $w.f.l -side top
	pack $w.f.sv $w.f.ci $w.f.cs -side top -anchor w
	pack $w.f -side top
    }
}
