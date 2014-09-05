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
#    File   : UnuCmedian.tcl
#    Author : Martin Cole
#    Date   : Mon Aug 25 10:14:23 2003

catch {rename Teem_Unu_UnuCmedian ""}

itcl_class Teem_Unu_UnuCmedian {
    inherit Module
    constructor {config} {
        set name UnuCmedian
        set_defaults
    }
    method set_defaults {} {
	global $this-radius
	global $this-weight
	global $this-bins
	global $this-pad
	

	set $this-radius 1
	set $this-weight 1.0
	set $this-bins 2048
	set $this-mode 0
	set $this-pad 0
    }

    method valid_int {new} {
	if {! [regexp "\\A\\d*\\Z" $new]} {
	    return 0
	}
	return 1
    }

    method valid_float {new} {
	if {! [regexp "\\A\\d*\\.?\\d*\\Z" $new]} {
	    return 0
	}
	return 1
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

	#radius
	iwidgets::entryfield $w.f.options.radius -labeltext "Radius:" \
	    -validate "$this valid_int %P"\
	    -textvariable $this-radius
	#weight
	iwidgets::entryfield $w.f.options.weight -labeltext "Weight:" \
	    -validate "$this valid_float %P"\
	    -textvariable $this-weight
	#bins
	iwidgets::entryfield $w.f.options.bins -labeltext "Bins:" \
	    -validate "$this valid_int %P"\
	    -textvariable $this-bins
	#pad
	label $w.f.options.padlabel -text "Pad:"
	checkbutton $w.f.options.pad -variable $this-pad

	#pad
	label $w.f.options.modelabel -text "Use Mode Filtering:"
	checkbutton $w.f.options.mode -variable $this-mode	

	pack $w.f.options.radius $w.f.options.weight $w.f.options.bins -side top -expand yes -fill x
	pack $w.f.options.modelabel $w.f.options.mode -side left -anchor w 
	pack $w.f.options.padlabel $w.f.options.pad -side left -anchor w 
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
