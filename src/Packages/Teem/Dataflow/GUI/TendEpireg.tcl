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

catch {rename Teem_Tend_TendEpireg ""}

itcl_class Teem_Tend_TendEpireg {
    inherit Module
    constructor {config} {
        set name TendEpireg
        set_defaults
    }
    method set_defaults {} {
        global this-gradient_list
        set this-gradient_list ""

        global this-reference
        set this-reference "-1"

        global this-blur_x
        set this-blur_x 1.0

        global this-blur_y
        set this-blur_y 2.0

        global this-threshold
        set this-threshold 0.0

        global this-cc_analysis
        set this-cc_analysis 1

        global this-fitting
        set this-fitting 0.70

        global this-kernel
        set this-kernel "cubicCR"

        global this-sigma
        set this-sigma 0.0

	global this-extent
	set this-extent 0.5
    }

    method update_text {} {
	set w .ui[modname]
	set $this-gradient_list [$w.f.options.gradient_list get 1.0 end]
    }

    method send_text {} {
	$this update_text
	$this-c needexecute
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

	option add *textBackground white	
	iwidgets::scrolledtext $w.f.options.gradient_list \
	    -vscrollmode dynamic -labeltext "List of gradients. example: (one gradient per line) 0.5645 0.32324 0.4432454"

	bind $w.f.options.gradient_list <Leave> "$this update_text"
	catch {$w.f.options.gradient_list insert end [set $this-gradient_list]}

        pack $w.f.options.gradient_list -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.reference \
	    -labeltext "reference:" -textvariable $this-reference
        pack $w.f.options.reference -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.blur_x -labeltext "blur_x:" \
	    -textvariable $this-blur_x
        pack $w.f.options.blur_x -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.blur_y -labeltext "blur_y:" \
	    -textvariable $this-blur_y
        pack $w.f.options.blur_y -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.threshold -labeltext "threshold:" \
	    -textvariable $this-threshold
        pack $w.f.options.threshold -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.cc_analysis \
	    -labeltext "cc_analysis:" -textvariable $this-cc_analysis
        pack $w.f.options.cc_analysis -side top -expand yes -fill x
        iwidgets::entryfield $w.f.options.fitting -labeltext "fitting:" \
	    -textvariable $this-fitting
        pack $w.f.options.fitting -side top -expand yes -fill x

	make_labeled_radio $w.f.options.kernel "Kernel:" "" \
		top $this-kernel \
		{{"Box" box} \
		{"Tent" tent} \
		{"Cubic (Catmull-Rom)" cubicCR} \
		{"Cubic (B-spline)" cubicBS} \
		{"Quartic" quartic} \
		{"Gaussian" gaussian}}

        pack $w.f.options.kernel -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.sigma -labeltext "sigma:" \
	    -textvariable $this-sigma
        pack $w.f.options.sigma -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.extent -labeltext "extent:" \
	    -textvariable $this-extent
        pack $w.f.options.extent -side top -expand yes -fill x

	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
