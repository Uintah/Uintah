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


itcl_class SCIRun_FieldsCreate_SampleStructHex {
    inherit Module
    constructor {config} {
        set name SampleStructHex
        set_defaults
    }

    method set_defaults {} {
	global $this-sizex
	global $this-sizey
	global $this-sizez
	global $this-padpercent
	global $this-data-at
	set $this-sizex 16
	set $this-sizey 16
	set $this-sizez 16
	set $this-padpercent 0
	set $this-data-at Nodes
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
	frame $w.row31
	frame $w.which -relief groove -borderwidth 2

	pack $w.row1 $w.row2 $w.row3 $w.row31 $w.which -side top \
		-e y -f both -padx 5 -pady 5
	
	label $w.row1.xsize_label -text "X Size   "
	entry $w.row1.xsize -textvariable $this-sizex
	label $w.row2.ysize_label -text "Y Size   "
	entry $w.row2.ysize -textvariable $this-sizey
	label $w.row3.zsize_label -text "Z Size   "
	entry $w.row3.zsize -textvariable $this-sizez

	label $w.row31.zsize_label -text "Pad Percentage"
	entry $w.row31.zsize -textvariable $this-padpercent

	pack $w.row1.xsize_label $w.row1.xsize -side left
	pack $w.row2.ysize_label $w.row2.ysize -side left
	pack $w.row3.zsize_label $w.row3.zsize -side left
	pack $w.row31.zsize_label $w.row31.zsize -side left

	label $w.which.l -text "Data at Location"
	radiobutton $w.which.node -text "Nodes (linear basis)" \
		-variable $this-data-at -value Nodes
	radiobutton $w.which.cell -text "Cells (constant basis)" \
		-variable $this-data-at -value Cells
	radiobutton $w.which.none -text "None" \
		-variable $this-data-at -value None
	pack $w.which.l -side top
	pack $w.which.node $w.which.cell $w.which.none -anchor nw

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


