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
 #  PlanarContoursToSegVol.tcl: Turn a contour set into a seg field
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

catch {rename BioPSE_Modeling_PlanarContoursToSegVol ""}

itcl_class DaveW_FEM_PlanarContoursToSegVol {
    inherit Module
    constructor {config} {
        set name PlanarContoursToSegVol
        set_defaults
    }
    method set_defaults {} {
	global $this-nxTCL
	set $this-nxTCL "16"
	global $this-nyTCL
	set $this-nyTCL "16"
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
	global $this-nxTCL
	global $this-nyTCL
	frame $w.f.nx
	label $w.f.nx.l -text "NX: "
	entry $w.f.nx.e -relief sunken -width 4 -textvariable $this-nxTCL
	pack $w.f.nx.l $w.f.nx.e -side left
	frame $w.f.ny
	label $w.f.ny.l -text "NY: "
	entry $w.f.ny.e -relief sunken -width 4 -textvariable $this-nyTCL
	pack $w.f.ny.l $w.f.ny.e -side left
	pack $w.f.nx $w.f.ny -side top
        pack $w.f -side top -expand yes
    }
}
