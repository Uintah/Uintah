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


itcl_class SCIRun_FieldsOther_FieldCage {
    inherit Module
    constructor {config} {
        set name FieldCage
        set_defaults
    }

    method set_defaults {} {
	global $this-sizex
	global $this-sizey
	global $this-sizez
	set $this-sizex 10
	set $this-sizey 10
	set $this-sizez 10
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.row1
	frame $w.row2
	frame $w.row3
	frame $w.row4

	pack $w.row1 $w.row2 $w.row3 $w.row4 -side top \
		-e y -f both -padx 5 -pady 5
	
	label $w.row1.xsize_label -text "X Size   "
	Tooltip $w.row1.xsize_label "Number of lines in the X direction (Red lines)"
	entry $w.row1.xsize -textvariable $this-sizex

	label $w.row2.ysize_label -text "Y Size   "
	Tooltip $w.row2.ysize_label "Number of lines in the Y direction (Green lines)"
	entry $w.row2.ysize -textvariable $this-sizey

	label $w.row3.zsize_label -text "Z Size   "
	Tooltip $w.row3.zsize_label "Number of lines in the Z direction (Blue lines)"
	entry $w.row3.zsize -textvariable $this-sizez

	bind $w.row1.xsize <KeyPress-Return> "$this-c needexecute"
	bind $w.row2.ysize <KeyPress-Return> "$this-c needexecute"
	bind $w.row3.zsize <KeyPress-Return> "$this-c needexecute"

	bind $w.row1.xsize <KeyPress-Tab> "$this-c needexecute"
	bind $w.row2.ysize <KeyPress-Tab> "$this-c needexecute"
	bind $w.row3.zsize <KeyPress-Tab> "$this-c needexecute"

	pack $w.row1.xsize_label $w.row1.xsize -side left
	pack $w.row2.ysize_label $w.row2.ysize -side left
	pack $w.row3.zsize_label $w.row3.zsize -side left

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


