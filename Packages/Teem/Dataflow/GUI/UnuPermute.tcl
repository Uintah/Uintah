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
 #  UnuPermute.tcl: The UnuPermute UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Unu_UnuPermute ""}

itcl_class Teem_Unu_UnuPermute {
    inherit Module
    constructor {config} {
        set name UnuPermute
        set_defaults
    }
    method set_defaults {} {
        global $this-dim

        set $this-dim 0
    }

    # never permute the tuple axis (axis 0) so no interface for axis 0
    method make_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    for {set i 1} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.a$i]} {
		    make_entry $w.f.a$i "Axis $i <- " $this-axis$i
		    pack $w.f.a$i -side top -expand 1 -fill x
		}
	    }
	}
    }
    
    # never permute the tuple axis (axis 0)
    method init_axes {} {
	for {set i 0} {$i < [set $this-dim]} {incr i} {
	    #puts "init_axes----$i"

	    if { [catch { set t [set $this-axis$i] } ] } {
		set $this-axis$i $i
	    }
	}
	make_axes
    }

    method clear_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    for {set i 1} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {[winfo exists $w.f.a$i]} {
		    destroy $w.f.a$i
		}
		unset $this-axis$i
	    }
	}
    }

    method make_entry {w text v} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        global $v
        entry $w.e -textvariable $v -width 2
        pack $w.e -side right
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 80
        frame $w.f

	if {[set $this-dim] == 0} {
	    label $w.f.t -text "Need Execute to know the number of Axes."
	    pack $w.f.t
	} else {
	    init_axes 
	}

	pack $w.f

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
