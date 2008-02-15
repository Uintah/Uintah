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


# GUI for IsoValueController module
# by Allen R. Sanderson
# SCI Institute
# University of Utah
# September 2005

catch {rename Fusion_Fields_StreamlineAnalyzer ""}

itcl_class Fusion_Fields_StreamlineAnalyzer {
    inherit Module
    constructor {config} {
        set name StreamlineAnalyzer
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

	toplevel $w

	frame $w.planes -relief groove -borderwidth 2

	frame $w.planes.list
	label $w.planes.list.l -text "List of Planes:"
	entry $w.planes.list.e -width 20 -text $this-planes-list
	bind $w.planes.list.e <Return> "$this-c needexecute"
	pack $w.planes.list.l $w.planes.list.e -side left -expand 1

	###### Save the plane-quantity because the iwidget resets it
	frame $w.planes.quant
	global $this-planes-quantity
	set quantity [set $this-planes-quantity]
	iwidgets::spinint $w.planes.quant.q \
	    -labeltext "Number of evenly-spaced planes: " \
	    -range {0 100} -step 1 \
	    -textvariable $this-planes-quantity \
	    -width 10 -fixed 10 -justify right
	
	$w.planes.quant.q delete 0 end
	$w.planes.quant.q insert 0 $quantity

	pack $w.planes.quant.q -side top -expand 1

	pack $w.planes.list $w.planes.quant -side left -fill x


	frame $w.color -relief groove -borderwidth 2
	label $w.color.label -text "Color Style"

	frame $w.color.left
	radiobutton $w.color.left.orig     -text "Original Value" \
	    -variable $this-color -value 0
	radiobutton $w.color.left.input    -text "Input Order" \
	    -variable $this-color -value 1
	radiobutton $w.color.left.index    -text "Point Index" \
	    -variable $this-color -value 2

	frame $w.color.middle
	radiobutton $w.color.middle.plane   -text "Plane" \
	    -variable $this-color -value 3
	radiobutton $w.color.middle.winding -text "Winding Group" \
	    -variable $this-color -value 4
	radiobutton $w.color.middle.order   -text "Winding Group Order" \
	    -variable $this-color -value 5

	frame $w.color.right
	radiobutton $w.color.right.winding -text "Number of Windings" \
	    -variable $this-color -value 6
	radiobutton $w.color.right.twist   -text "Number of Twists" \
	    -variable $this-color -value 7
	radiobutton $w.color.right.safety  -text "Saftey Factor" \
	    -variable $this-color -value 8

	pack $w.color.left.orig $w.color.left.input \
	    $w.color.left.index -side top -anchor w
	pack $w.color.middle.plane $w.color.middle.winding \
	    $w.color.middle.order -side top -anchor w
	pack $w.color.right.winding $w.color.right.twist \
	    $w.color.right.safety -side top -anchor w

	pack $w.color.label -side top -fill both
	pack $w.color.left $w.color.middle $w.color.right \
	    -side left -anchor w


	frame $w.windings -relief groove -borderwidth 2

#	checkbutton $w.windings.check -variable $this-override-check

	frame $w.windings.values

	global $this-override
	set override [set $this-override]
	iwidgets::spinint $w.windings.values.override \
	    -labeltext "Override winding: " \
	    -range {0 1000} -step 1 \
	    -textvariable $this-override \
	    -width 10 -fixed 10 -justify right
	
	$w.windings.values.override delete 0 end
	$w.windings.values.override insert 0 $override


	global $this-maxWindings
	set max [set $this-maxWindings]
	iwidgets::spinint $w.windings.values.max \
	    -labeltext "    Max windings: " \
	    -range {0 1000} -step 1 \
	    -textvariable $this-maxWindings \
	    -width 10 -fixed 10 -justify right
	
	$w.windings.values.max delete 0 end
	$w.windings.values.max insert 0 $max

	pack $w.windings.values.max $w.windings.values.override -side top


	frame $w.windings.order
	label $w.windings.order.label -text "  Order Preference"
	radiobutton $w.windings.order.low -text "Low" \
	    -variable $this-order -value 0

	radiobutton $w.windings.order.high -text "High" \
	    -variable $this-order -value 1

	radiobutton $w.windings.order.node -text "Node" \
	    -variable $this-order -value 2

	pack $w.windings.order.label \
	    $w.windings.order.low $w.windings.order.high \
	    $w.windings.order.node \
	    -side left -anchor w

	pack $w.windings.values $w.windings.order \
	    -side left -fill x


	frame $w.mesh -relief groove -borderwidth 2
	radiobutton $w.mesh.crv -text "Curve Mesh   " \
	    -variable $this-curve-mesh -value 1

	radiobutton $w.mesh.srf -text "Surface Mesh     " \
	    -variable $this-curve-mesh -value 0


	frame $w.mesh.island
	radiobutton $w.mesh.island.centroids -text "Island Centroids" \
	    -variable $this-island-centroids -value 1

	radiobutton $w.mesh.island.nulls -text "Island Nulls    " \
	    -variable $this-island-centroids -value 0

	pack $w.mesh.island.centroids $w.mesh.island.nulls -side left -anchor w

	pack $w.mesh.crv $w.mesh.srf $w.mesh.island -side left -anchor w


	frame $w.field -relief groove -borderwidth 2
	radiobutton $w.field.scalar -text "Scalar Field" \
	    -variable $this-scalar-field -value 1

	radiobutton $w.field.vector -text "Vector Field" \
	    -variable $this-scalar-field -value 0

	pack $w.field.scalar $w.field.vector -side left -anchor w


	frame $w.misc -relief groove -borderwidth 2

	frame $w.misc.islands
	checkbutton $w.misc.islands.check -variable $this-show-islands
	label $w.misc.islands.label -text "Show Only Islands" -width 18 \
	    -anchor w -just left
	pack $w.misc.islands.check $w.misc.islands.label -side left


	frame $w.misc.overlaps
	label $w.misc.overlaps.label -text "Overlaps" \
	    -width 9 -anchor w -just left

	radiobutton $w.misc.overlaps.raw -text "Raw" \
	    -variable $this-overlaps -value 0
	radiobutton $w.misc.overlaps.remove -text "Remove" \
	    -variable $this-overlaps -value 1
	radiobutton $w.misc.overlaps.merge -text "Merge" \
	    -variable $this-overlaps -value 2
	radiobutton $w.misc.overlaps.smooth -text "Smooth" \
	    -variable $this-overlaps -value 3

	pack $w.misc.overlaps.label $w.misc.overlaps.raw \
	    $w.misc.overlaps.remove $w.misc.overlaps.merge \
	    $w.misc.overlaps.smooth -side left -anchor w


	pack $w.misc.islands $w.misc.overlaps \
	    -side left -anchor w


	pack $w.planes $w.color $w.windings $w.mesh $w.field $w.misc \
	    -side top -padx 4 -pady 4 -fill both

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
