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

#    File   : UnuCrop.tcl
#    Author : Martin Cole
#    Date   : Tue Mar 18 08:46:53 2003

catch {rename Teem_Unu_UnuCrop ""}

itcl_class Teem_Unu_UnuCrop {
    inherit Module
    constructor {config} {
        set name UnuCrop
        set_defaults
    }
    method set_defaults {} {
	global $this-num-axes
	global $this-reset

	set $this-num-axes 0
	set $this-reset 0
    }

    method set_max_vals {} {
	set w .ui[modname]

	if {[winfo exists $w]} {
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {

		set ax $this-absmaxAxis$i
		set_scale_max_value $w.f.mmf.a$i [set $ax]
		if {[set $this-maxAxis$i] == 0 || [set $this-reset]} {
		    set $this-maxAxis$i [set $this-absmaxAxis$i]
		}
	    }
	    set $this-reset 0
	} else {
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {

		if {[set $this-maxAxis$i] == 0 || [set $this-reset]} {
		    set $this-maxAxis$i [set $this-absmaxAxis$i]
		}
	    }
	    set $this-reset 0
	}
    }

    method init_axes {} {
	for {set i 0} {$i < [set $this-num-axes]} {incr i} {
	    #puts "init_axes----$i"

	    if { [catch { set t [set $this-minAxis$i] } ] } {
		set $this-minAxis$i 0
		#puts "made minAxis$i"
	    }
	    if { [catch { set t [set $this-maxAxis$i]}] } {
		set $this-maxAxis$i 0
		#puts "made maxAxis$i   [set $this-maxAxis$i]"
	    }
	    if { [catch { set t [set $this-absmaxAxis$i]}] } {
		set $this-absmaxAxis$i 0
		#puts "made absmaxAxis$i"
	    }
	}
	make_min_max
    }

    method make_min_max {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
  
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {
		#puts $i
		if {! [winfo exists $w.f.mmf.a$i]} {
		    min_max_widget $w.f.mmf.a$i "Axis $i" \
			$this-minAxis$i $this-maxAxis$i $this-absmaxAxis$i 
		}
	    }
	}
    }

   method clear_axes {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    if {[winfo exists $w.f.mmf.t]} {
		destroy $w.f.mmf.t
	    }
	    for {set i 0} {$i < [set $this-num-axes]} {incr i} {
		#puts $i
		if {[winfo exists $w.f.mmf.a$i]} {
		    destroy $w.f.mmf.a$i
		}
		unset $this-minAxis$i
		unset $this-maxAxis$i
		unset $this-absmaxAxis$i
	    }
	    set $this-reset 1
	}
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        wm minsize $w 150 80
        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.mmf
	pack $w.f.mmf -padx 2 -pady 2 -side top -expand yes
	
	if {[set $this-num-axes] == 0} {
	    label $w.f.mmf.t -text "Need to Execute to know the number of Axes."
	    pack $w.f.mmf.t
	} else {
	    init_axes 
	}

        makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
