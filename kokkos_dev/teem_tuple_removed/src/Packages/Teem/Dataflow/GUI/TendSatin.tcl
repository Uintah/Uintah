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
#    File   : TendSatin.tcl
#    Author : Darby Van Uitert
#    Date   : April 2004

itcl_class Teem_Tend_TendSatin {
    inherit Module
    constructor {config} {
        set name TendSatin
        set_defaults
    }

    method set_defaults {} {
	global $this-torus
	set $this-torus 0

	global $this-anisotropy
	set $this-anisotropy 1.0

	global $this-maxca1
	set $this-maxca1 1.0

	global $this-minca1
	set $this-minca1 0.0

	global $this-boundary
	set $this-boundary 0.05
	
	global $this-thickness
	set $this-thickness {0.3}

	global $this-size
	set $this-size {32}
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

	checkbutton $w.f.options.torus \
	    -text "Generate Torus Dataset" \
	    -variable $this-torus
	pack $w.f.options.torus -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.anisotropy \
	    -labeltext "Anisotropy Parameter:" \
	    -textvariable $this-anisotropy
        pack $w.f.options.anisotropy -side top -expand yes -fill x
	

        iwidgets::entryfield $w.f.options.maxca1 \
	    -labeltext "Max Anisotropy:" \
	    -textvariable $this-maxca1
        pack $w.f.options.maxca1 -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.minca1 \
	    -labeltext "Min Anisotropy:" \
	    -textvariable $this-minca1
        pack $w.f.options.minca1 -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.boundary \
	    -labeltext "Boundary:" \
	    -textvariable $this-boundary
        pack $w.f.options.boundary -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.thickness \
	    -labeltext "Thickness:" \
	    -textvariable $this-thickness
        pack $w.f.options.thickness -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.size \
	    -labeltext "Size:" \
	    -textvariable $this-size
        pack $w.f.options.size -side top -expand yes -fill x

	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
