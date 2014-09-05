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

#    File   : UnuJoin.tcl
#    Author : Martin Cole
#    Date   : Thu Jan 16 09:44:07 2003

itcl_class Teem_UnuAtoM_UnuJoin {
    inherit Module
    constructor {config} {
        set name UnuJoin
        set_defaults
    }

    method set_defaults {} {
	global $this-dim
	global $this-join-axis
	global $this-incr-dim

	set $this-dim 0
	set $this-join-axis 0
	set $this-incr-dim 0
    }
   
    # do not allow spaces in the label
    method valid_string {ind str} {
	set char "a"
	
	set char [string index $str $ind]
	if {$ind >= 0 && [string equal $char " "]} {
	    return 0
	}
	return 1
    }


    method axis_radio {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    if {[winfo exists $w.f.rfr.radio]} { destroy $w.f.rfr.radio }

	    set choices [list]

	    for {set i 0} {$i < [set $this-dim]} {incr i} {
		set lab "Axis $i"
		lappend choices [list $lab $i]
	    }

	    make_labeled_radio $w.f.rfr.radio \
		"Join Axis"  "" top $this-join-axis $choices		
	    pack $w.f.rfr.radio -fill both -expand 1 -side top
        }	
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.f -borderwidth 2
	pack $w.f -side top -e y -f both -padx 5 -pady 5

	#frame to pack and repack radio button in
	frame $w.f.rfr -relief groove -borderwidth 2

	axis_radio

	checkbutton $w.f.incrdim \
		-text "Increment Dimension" \
		-variable $this-incr-dim
	
	pack $w.f.rfr $w.f.incrdim -fill both -expand 1 -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	# button $w.execute -text "Ok" -command "destroy $w"
	# pack $w.execute -side top -e n -f both
    }
}


