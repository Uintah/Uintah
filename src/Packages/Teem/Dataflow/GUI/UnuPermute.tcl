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

    method make_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    for {set i 0} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.a$i]} {
		    make_entry $w.f.a$i "Axis $i <- " $this-axis$i
		    pack $w.f.a$i -side top -expand 1 -fill x
		}
	    }
	}
    }
    
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
	    for {set i 0} {$i < [set $this-dim]} {incr i} {
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
            return
        }

        toplevel $w
        wm minsize $w 100 80
        frame $w.f

	if {[set $this-dim] == 0} {
	    label $w.f.t -text "Need to Execute to know the number of Axes."
	    pack $w.f.t
	} else {
	    init_axes 
	}

	pack $w.f -side top -anchor nw -padx 3 -pady 3

        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
