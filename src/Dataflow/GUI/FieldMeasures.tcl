#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
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
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
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
	pack $w.general.l -side top
	pack $w.general.x $w.general.y $w.general.z $w.general.idx $w.general.nnbrs $w.general.size -anchor nw

	pack $w.which $w.general -side top -fill x -expand 1
    }
}
