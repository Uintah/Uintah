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
 #  UnuQuantize.tcl: The UnuQuantize UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Unu_UnuQuantize ""}

itcl_class Teem_Unu_UnuQuantize {
    inherit Module
    constructor {config} {
        set name UnuQuantize
        set_defaults
    }
    method set_defaults {} {
        global $this-minf
        global $this-maxf
        global $this-nbits
        set $this-minf 0
        set $this-maxf 255
	set $this-nbits 32
    }

    method update_min_max {min max} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    puts $min
	    puts $max
	    $w.f.min newvalue $min
	    $w.f.max newvalue $max
	}
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
	
        toplevel $w
        wm minsize $w 200 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
	global $this-minf
	expscale $w.f.min -orient horizontal -label "Min:" \
	        -variable $this-minf -command ""
	global $this-maxf
	expscale $w.f.max -orient horizontal -label "Max:" \
	        -variable $this-maxf -command ""
	global $this-nbits
	make_labeled_radio $w.f.nbits "Number of bits:" "" \
	 		left $this-nbits \
			{{8 8} \
			{16 16} \
			{32 32}}
	button $w.f.b -text "Execute" -command "$this-c needexecute"
	pack $w.f.min $w.f.max $w.f.nbits $w.f.b -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
