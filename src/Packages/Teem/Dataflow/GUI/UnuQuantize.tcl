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
 #  UnuQuantize.tcl: The UnuQuantize UI
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jan 2000
 #  Copyright (C) 2000 SCI Group
 ##

catch {rename Teem_UnuNtoZ_UnuQuantize ""}

itcl_class Teem_UnuNtoZ_UnuQuantize {
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
            return
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
	pack $w.f.min $w.f.max $w.f.nbits -side top -expand 1 -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}
