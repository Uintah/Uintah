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

# GUI for FusionSlicePlot module
# by Allen R. Sanderson
# March 2002

# This GUI interface consists of a widget that allows for scaling of 
# a height field in a surface.

itcl_class Fusion_Fields_FusionSlicePlot {
    inherit Module
    constructor {config} {
        set name FusionSlicePlot
        set_defaults
    }

    method set_defaults {} {

	global $this-scale
	set $this-scale 1.00
    }

    method ui {} {

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w

	expscale $w.slide -label Scale \
		-orient horizontal \
		-variable $this-scale -command "$this-c needexecute"

	bind $w.slide.scale <ButtonRelease> \
		"$this-c needexecute"

    }
}