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
#    File   : TendEpireg.tcl
#    Author : Martin Cole
#    Date   : Mon Sep  8 09:46:23 2003

catch {rename Teem_Tensor_TendEpireg ""}

itcl_class Teem_Tensor_TendEpireg {
    inherit Module
    constructor {config} {
        set name TendEpireg
        set_defaults
    }
    method set_defaults {} {
        global dwi_list
        set dwi_list ""

        global gradient_list
        set gradient_list ""

        global reference
        set reference 0

        global blur_x
        set blur_x 0.0

        global blur_y
        set blur_y 0.0

        global threshold
        set threshold 0.0

        global cc_analysis
        set cc_analysis 0

        global fitting
        set fitting 0.0

        global kernel
        set kernel ""

        global base
        set base 0


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

        iwidgets::entryfield $w.f.options.dwi_list -labeltext "dwi_list:" -textvariable $this-dwi_list
        pack $w.f.options.dwi_list -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.gradient_list -labeltext "gradient_list:" -textvariable $this-gradient_list
        pack $w.f.options.gradient_list -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.reference -labeltext "reference:" -textvariable $this-reference
        pack $w.f.options.reference -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.blur_x -labeltext "blur_x:" -textvariable $this-blur_x
        pack $w.f.options.blur_x -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.blur_y -labeltext "blur_y:" -textvariable $this-blur_y
        pack $w.f.options.blur_y -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.threshold -labeltext "threshold:" -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.cc_analysis -labeltext "cc_analysis:" -textvariable $this-cc_analysis
        pack $w.f.options.cc_analysis -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.fitting -labeltext "fitting:" -textvariable $this-fitting
        pack $w.f.options.fitting -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.kernel -labeltext "kernel:" -textvariable $this-kernel
        pack $w.f.options.kernel -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.base -labeltext "base:" -textvariable $this-base
        pack $w.f.options.base -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
