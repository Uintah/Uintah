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
#    File   : TendAnhist.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendAnhist {
    inherit Module
    constructor {config} {
        set name TendAnhist
        set_defaults
    }

    method set_defaults {} {
	global $this-westin
	set $this-westin 1

	global $this-resolution
	set $this-resolution 256
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

       iwidgets::entryfield $w.f.options.westin \
	    -labeltext "Version of Westin's Anisotropy\nMetric Triple: (1 or 2):" \
	    -textvariable $this-westin
        pack $w.f.options.westin -side top -expand yes -fill x
	

        iwidgets::entryfield $w.f.options.resolution \
	    -labeltext "Resolution:" \
	    -textvariable $this-resolution
        pack $w.f.options.resolution -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


