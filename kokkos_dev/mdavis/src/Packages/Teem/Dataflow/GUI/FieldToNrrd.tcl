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

#    File   : FieldToNrrd.tcl
#    Author : Martin Cole
#    Date   : Thu Jan 16 09:44:07 2003

itcl_class Teem_Converters_FieldToNrrd {
    inherit Module
    constructor {config} {
        set name FieldToNrrd
        set_defaults
    }

    method set_defaults {} {
	global $this-label
	set $this-label "unknown"
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

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.lab -relief groove -borderwidth 2

	pack $w.lab -side top -e y -f both -padx 5 -pady 5
	
	label $w.lab.label -text "Label Data:"
	entry $w.lab.dat-label -textvariable $this-label \
	    -validate key -validatecommand "$this valid_string %i %P"

	pack $w.lab.label $w.lab.dat-label -side left -padx 4 -pady 4

	button $w.execute -text "Set Label:" -command "wm withdraw $w"
	pack $w.execute -side top -e n -f both -padx 4 -pady 2

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


