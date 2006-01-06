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
 #  UnuPad.tcl: The UnuPad UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_UnuNtoZ_UnuPad ""}

itcl_class Teem_UnuNtoZ_UnuPad {
    inherit Module
    constructor {config} {
        set name UnuPad
        set_defaults
    }
    method set_defaults {} {
	global $this-pad-style
	global $this-pad-value

	global $this-dim

	set $this-pad-style Bleed
	set $this-pad-value 0

	set $this-dim 0
    }

    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.t]} {
		destroy $w.f.t
	    }
	    for {set i 0} {$i < [set $this-dim]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.a$i]} {
		    add_axis $w.f.a$i $i $this-minAxis$i $this-maxAxis$i
		    pack $w.f.a$i -side top -expand 1 -fill x
		}
	    }
	}
    }
    
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
	    for {set i 0} {$i < [set $this-dim]} {incr i} {
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

	frame $w.style
	radiobutton $w.style.bleed -text "Bleed" \
	    -variable $this-pad-style -value "Bleed"
	radiobutton $w.style.wrap -text "Wrap" \
	    -variable $this-pad-style -value "Wrap"
	radiobutton $w.style.pad -text "Pad" \
	    -variable $this-pad-style -value "Pad"

        label $w.style.l -text "Value"
        entry $w.style.v -textvariable $this-pad-value -width 6

	pack $w.style.bleed $w.style.wrap $w.style.pad $w.style.l $w.style.v \
	    -side left -anchor nw -padx 3

	pack $w.style -side top

        frame $w.f
	frame $w.fb
        pack $w.f $w.fb -padx 2 -pady 2 -side top -expand yes


	if {[set $this-dim] == 0} {
	    label $w.f.t -text "Need to Execute to know the number of Axes."
	    pack $w.f.t
	} else {
	    init_axes 
	}

        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
