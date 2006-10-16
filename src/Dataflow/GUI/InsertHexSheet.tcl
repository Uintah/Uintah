#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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
 #  InsertHexSheet.tcl: The InsertHexSheet UI
 #  Written by:
 #   Jason Shepherd
 #   Department of Computer Science
 #   University of Utah
 #   April 2006
 #  Copyright (C) 2006 SCI Group
 ##

catch {rename SCIRun_FieldsCreate_InsertHexSheet ""}

itcl_class SCIRun_FieldsCreate_InsertHexSheet {
    inherit Module

    constructor {config} {
        set name InsertHexSheet
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 80

	frame $w.bound1
	label $w.bound1.t1 -text "Intersected Hexes in"
	pack $w.bound1.t1
	pack $w.bound1

	frame $w.bound
	radiobutton $w.bound.side_1 -text "Side 1" \
	    -variable $this-side -value "side1"
	radiobutton $w.bound.side_2 -text "Side 2" \
	    -variable $this-side -value "side2"
	pack $w.bound.side_1 $w.bound.side_2 \
	    -side left -anchor nw -padx 3
	pack $w.bound -side top

	frame $w.layer1
	label $w.layer1.t1
	label $w.layer1.t2 -text "Add Sheet?"
	pack $w.layer1.t1 $w.layer1.t2
	pack $w.layer1

	frame $w.layer
	radiobutton $w.layer.addlayeron -text "On" \
	    -variable $this-addlayer -value "On"
	radiobutton $w.layer.addlayeroff -text "Off" \
	    -variable $this-addlayer -value "Off"
	pack $w.layer.addlayeron $w.layer.addlayeroff \
	    -side left -anchor nw -padx 3
	pack $w.layer -side top

        frame $w.f
 	frame $w.fb
	pack $w.f $w.fb -padx 2 -pady 2 -side top -expand yes

        makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
