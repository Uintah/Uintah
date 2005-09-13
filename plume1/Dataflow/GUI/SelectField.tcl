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


itcl_class SCIRun_FieldsOther_SelectField {
    inherit Module
    constructor {config} {
        set name SelectField

	global $this-stampvalue
	global $this-runmode

        set_defaults
    }

    method set_defaults {} {
	set $this-stampvalue 100
	set $this-runmode 0
	# 0 nothing 1 accumulate 2 replace
    }

    method replace {} {
	set $this-runmode 2
	$this-c needexecute
    }

    method accumulate {} {
	set $this-runmode 1
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.row1
	frame $w.row3
	frame $w.row4

	pack $w.row1 $w.row3 $w.row4 -side top -e y -f both -padx 5 -pady 5
	
	label $w.row1.value_label -text "Selection Value"
	entry $w.row1.value -textvariable $this-stampvalue
	pack $w.row1.value_label $w.row1.value -side left

	button $w.row3.execute -text "Replace" -command "$this replace"
	pack $w.row3.execute -side top -e n -f both

	button $w.row4.execute -text "Accumulate" -command "$this accumulate"
	pack $w.row4.execute -side top -e n -f both

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


