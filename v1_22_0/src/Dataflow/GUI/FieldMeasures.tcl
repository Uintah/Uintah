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


itcl_class SCIRun_FieldsOther_FieldMeasures {
    inherit Module
    constructor {config} {
        set name FieldMeasures

	global $this-simplexString
	global $this-xFlag
	global $this-yFlag
	global $this-zFlag
	global $this-idxFlag
	global $this-sizeFlag
	global $this-numNbrsFlag
	global $this-normalsFlag
        set_defaults
    }

    method set_defaults {} {
	set $this-simplexString Node
	set $this-xFlag 1
	set $this-yFlag 1
	set $this-zFlag 1
	set $this-idxFlag 0
	set $this-sizeFlag 0
	set $this-numNbrsFlag 0
	set $this-normalsFlag 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.which -relief groove -borderwidth 2
	label $w.which.l -text "Measure Locations"
	radiobutton $w.which.node -text "Nodes" \
		-var $this-simplexString -value Node
	radiobutton $w.which.edge -text "Edges" \
		-var $this-simplexString -value Edge
	radiobutton $w.which.face -text "Faces" \
		-var $this-simplexString -value Face
	radiobutton $w.which.cell -text "Cells" \
		-var $this-simplexString -value Cell
	radiobutton $w.which.elem -text "Elements" \
		-var $this-simplexString -value Elem
	pack $w.which.l -side top
	pack $w.which.elem $w.which.node $w.which.edge $w.which.face $w.which.cell -anchor nw

	frame $w.general -relief groove -borderwidth 2
	label $w.general.l -text "Measures"
	checkbutton $w.general.x -text "X position" -variable $this-xFlag
	checkbutton $w.general.y -text "Y position" -variable $this-yFlag
	checkbutton $w.general.z -text "Z position" -variable $this-zFlag
	checkbutton $w.general.idx -text "Index" -variable $this-idxFlag
	checkbutton $w.general.nnbrs -text "Valence" -variable $this-numNbrsFlag
	checkbutton $w.general.size -text "Size (Length, Area, or Volume)" -variable $this-sizeFlag
	checkbutton $w.general.norm -text "Normals (Nodes and Faces only)" -variable $this-normalsFlag
	pack $w.general.l -side top
	pack $w.general.x $w.general.y $w.general.z $w.general.idx $w.general.nnbrs $w.general.size $w.general.norm -anchor nw

	pack $w.which $w.general -side top -fill x -expand 1

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
