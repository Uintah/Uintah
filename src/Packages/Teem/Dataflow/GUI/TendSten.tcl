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
#    File   : TendSten.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendSten {
    inherit Module
    constructor {config} {
        set name TendSten
        set_defaults
    }

    method set_defaults {} {
	global $this-diffscale
	set $this-diffscale 1

	global $this-intscale
	set $this-intscale 2

	global $this-factor
	set $this-factor 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.diffscale \
	    -labeltext "Difference Scale:" \
	    -textvariable $this-diffscale
        pack $w.f.options.diffscale -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.intscale \
	    -labeltext "Integration Scale:" \
	    -textvariable $this-intscale
        pack $w.f.options.intscale -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.factor \
	    -labeltext "Downsample Factor:" \
	    -textvariable $this-factor
        pack $w.f.options.factor -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}

