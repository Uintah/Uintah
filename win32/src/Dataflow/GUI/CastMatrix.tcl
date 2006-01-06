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


itcl_class SCIRun_Math_CastMatrix {
    inherit Module
    constructor {config} {
        set name CastMatrix
        set_defaults
    }
    method set_defaults {} {
        global $this-newtype
        global $this-oldtype
	global $this-nrow
	global $this-ncol
	set $this-newtype "Same"
        set $this-oldtype "Unknown"
	set $this-nrow "??"
	set $this-ncol "??"
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w
        wm minsize $w 150 20

	frame $w.i -relief ridge -borderwidth 2
	pack $w.i -side top -padx 2 -pady 2 -side top

	frame $w.i.size
        pack $w.i.size -side top
	global $this-nrow $this-ncol
	label $w.i.size.label -text "Input Matrix Size:   nrows ="
	label $w.i.size.row -textvariable $this-nrow
	label $w.i.size.x -text "  ncols ="
	label $w.i.size.col -textvariable $this-ncol
	pack $w.i.size.label $w.i.size.row $w.i.size.x $w.i.size.col -side left

        frame $w.i.type 
	pack $w.i.type -side top
        global $this-oldtype
	label $w.i.type.l -text "Input Matrix Type: " 
	label $w.i.type.v -textvariable $this-oldtype
	global $this-space
	label $w.i.type.s -textvariable $this-space
	pack $w.i.type.l $w.i.type.v $w.i.type.s -side left
	
	frame $w.otype -relief ridge -borderwidth 2
        pack $w.otype -side top -expand yes -padx 2 -pady 2 -fill x
        global $this-newtype
        make_labeled_radio $w.otype.r \
		"Output Matrix Type" "" \
                top $this-newtype \
		{{"Same (pass-through)" Same} \
		{ColumnMatrix ColumnMatrix} \
		{DenseMatrix DenseMatrix} \
		{SparseRowMatrix SparseRowMatrix} \
                {DenseColMajMatrix DenseColMajMatrix}}
	pack $w.otype.r -side top -expand 1 -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
