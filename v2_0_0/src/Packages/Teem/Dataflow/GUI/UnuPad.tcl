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
 #  UnuPad.tcl: The UnuPad UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_Unu_UnuPad ""}

itcl_class Teem_Unu_UnuPad {
    inherit Module
    constructor {config} {
        set name UnuPad
        set_defaults
    }
    method set_defaults {} {
	global $this-dim

	set $this-dim 0
    }

    # never pad the tuple axis (axis 0) so no interface for axis 0
    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    for {set i 1} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.a$i]} {
		    add_axis $w.f.a$i $i $this-minAxis$i $this-maxAxis$i
		    pack $w.f.a$i -side top -expand 1 -fill x
		}
	    }
	}
    }
    
    # never pad the tuple axis (axis 0)
    method init_axes {} {
	for {set i 0} {$i < [set $this-dim]} {incr i} {
	    #puts "init_axes----$i"

	    if { [catch { set t [set $this-minAxis$i] } ] } {
		set $this-minAxis$i 0
		#puts "made minAxis$i"
	    }
	    if { [catch { set t [set $this-maxAxis$i]}] } {
		set $this-maxAxis$i 0
		#puts "made maxAxis$i   [set $this-maxAxis$i]"
	    }
	}
	make_min_max
    }

    method clear_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 1} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {[winfo exists $w.f.mmf.a$i]} {
		    destroy $w.f.mmf.a$i
		}
		unset $this-minAxis$i
		unset $this-maxAxis$i
	    }
	    set $this-reset 1
	}
       }

    method add_axis {f axis_num vb va} {
	iwidgets::labeledframe $f -labeltext "Axis $axis_num" -labelpos n 
	set c [$f childsite]
	frame $c.f
	pack $c.f -side top -expand 1 -fill x
	set w $c.f
        label $w.lb -text "Prepend Pad"
        pack $w.lb -side left
        global $vb
        entry $w.eb -textvariable $vb -width 6
        pack $w.eb -side right

	frame $c.f1
	pack $c.f1 -side top -expand 1 -fill x
	set w $c.f1
	label $w.la -text "Append Pad"
        pack $w.la -side left
        global $va
        entry $w.ea -textvariable $va -width 6
        pack $w.ea -side right
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 80
        frame $w.f
	frame $w.fb
        pack $w.f $w.fb -padx 2 -pady 2 -side top -expand yes


	if {[set $this-dim] == 0} {
	    label $w.f.t -text "Need Execute to know the number of Axes."
	    pack $w.f.t
	} else {
	    init_axes 
	}

	button $w.fb.b -text "Execute" -command "$this-c needexecute"
	pack $w.fb.b -side top -expand 1 -fill x
    }
}
