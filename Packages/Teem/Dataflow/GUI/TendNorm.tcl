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
#    File   : TendEstim.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tend_TendNorm ""}

itcl_class Teem_Tend_TendNorm {
    inherit Module
    constructor {config} {
        set name TendNorm
        set_defaults
    }
    method set_defaults {} {
        global $this-major-weight
        set $this-major-weight 1.0

        global $this-medium-weight
        set $this-medium-weight 1.0

        global $this-minor-weight
        set $this-minor-weight 1.0

        global $this-amount
        set $this-amount 1.0

        global $this-target
        set $this-target 1.0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.major -labeltext "Major weight:" \
	    -textvariable $this-major-weight
        pack $w.f.options.major -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.medium -labeltext "Medium weight:" \
	    -textvariable $this-medium-weight
        pack $w.f.options.medium -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.minor -labeltext "Minor weight:" \
	    -textvariable $this-minor-weight
        pack $w.f.options.minor -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.amount -labeltext "Amount:" \
	    -textvariable $this-amount
	pack $w.f.options.amount -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.target -labeltext "Target:" \
	    -textvariable $this-target
        pack $w.f.options.target -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
