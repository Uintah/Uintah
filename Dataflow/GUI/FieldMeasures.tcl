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

itcl_class SCIRun_Fields_FieldMeasures {
    inherit Module
    constructor {config} {
        set name FieldMeasures

	global $this-nodeBased
	global $this-xFlag
	global $this-yFlag
	global $this-zFlag
	global $this-idxFlag
	global $this-sizeFlag
	global $this-valenceFlag
	global $this-lengthFlag
	global $this-aspectRatioFlag
	global $this-elemSizeFlag

        set_defaults
    }

    method set_defaults {} {
	set $this-nodeBased node
	set $this-xFlag 1
	set $this-yFlag 0
	set $this-zFlag 0
	set $this-idxFlag 0
	set $this-sizeFlag 0
	set $this-valenceFlag 0
	set $this-lengthFlag 0
	set $this-aspectRatioFlag 0
	set $this-elemSizeFlag 0
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
		-var $this-nodeBased -value node
	radiobutton $w.which.edge -text "Edges" \
		-var $this-nodeBased -value edge
	radiobutton $w.which.element -text "Elements" \
		-var $this-nodeBased -value element
	pack $w.which.l -side top
	pack $w.which.node $w.which.edge $w.which.element -anchor nw

	frame $w.general -relief groove -borderwidth 2
	label $w.general.l -text "General Measures"
	checkbutton $w.general.x -text "X position" -variable $this-xFlag
	checkbutton $w.general.y -text "Y position" -variable $this-yFlag
	checkbutton $w.general.z -text "Z position" -variable $this-zFlag
	checkbutton $w.general.idx -text "Index" -variable $this-idxFlag
	pack $w.general.l -side top
	pack $w.general.x $w.general.y $w.general.z $w.general.idx -anchor nw

	frame $w.node -relief groove -borderwidth 2
	label $w.node.l -text "Node Measures"
	checkbutton $w.node.valence -text "Valence" -variable $this-valenceFlag
	pack $w.node.l -side top
	pack $w.node.valence -anchor nw

	frame $w.edge -relief groove -borderwidth 2
	label $w.edge.l -text "Edge Measures"
	checkbutton $w.edge.length -text "Length" -variable $this-lengthFlag
	pack $w.edge.l -side top
	pack $w.edge.length -anchor nw

	frame $w.elem -relief groove -borderwidth 2
	label $w.elem.l -text "Element Measures"
	checkbutton $w.elem.aspect -text "Aspect Ratio" \
		-variable $this-aspectRatioFlag
	checkbutton $w.elem.size -text "Size (e.g. volume)" \
		-variable $this-elemSizeFlag
	pack $w.elem.l -side top
	pack $w.elem.aspect $w.elem.size -anchor nw

	pack $w.which $w.general $w.node $w.edge $w.elem -side top -fill x -expand 1
    }
}
