#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
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

catch {rename Teem_UnuNtoZ_UnuPermute ""}

itcl_class Teem_UnuNtoZ_UnuPermute {
    inherit Module
    constructor {config} {
        set name UnuPermute
        set_defaults
    }
    method set_defaults {} {
        global $this-dim
	global $this-uis
	global $this-axis0
	global $this-axis1
	global $this-axis2
	global $this-axix3

        set $this-dim 0
	set $this-uis 4
	set $this-axis0 0
	set $this-axis1 1
	set $this-axis2 2
	set $this-axis3 3
    }

    method make_axis {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    if {! [winfo exists $w.f.f.a$i]} {
		if {![info exists $this-axis$i]} {
		    set $this-axis$i $i
		}
		make_entry $w.f.f.a$i "Axis $i <- " $this-axis$i
		pack $w.f.f.a$i -side top -expand 1 -fill x
	    }
	}
    }
    

    method clear_axis {i} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    if {[winfo exists $w.f.f.a$i]} {
		destroy $w.f.f.a$i
	    }
	    unset $this-axis$i
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
            return
        }

        toplevel $w
        wm minsize $w 100 80
        frame $w.f

	frame $w.f.f

	# add permute axes
	for {set i 0} {$i < [set $this-uis]} {incr i} {
	    make_axis $i
	}

	pack $w.f.f -side top -expand 1 -fill x

	# add buttons to increment/decrement resample axes
	frame $w.f.buttons
	pack $w.f.buttons -side top -anchor n

	button $w.f.buttons.add -text "Add Axis" -command "$this-c add_axis"
	button $w.f.buttons.remove -text "Remove Axis" -command "$this-c remove_axis"
	
	pack $w.f.buttons.add $w.f.buttons.remove -side left -anchor nw -padx 5 -pady 5	
	
	pack $w.f -side top -anchor nw -padx 3 -pady 3

        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
