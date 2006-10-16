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


itcl_class SCIRun_FieldsCreate_ClipField {
    inherit Module

    constructor {config} {
        set name ClipField
    }

    method execrunmode {} {
	set $this-execmode execute
	$this-c needexecute
    }

    method invert {} {
	set $this-execmode invert
	$this-c needexecute
    }

    method undo {} {
	set $this-execmode undo
	$this-c needexecute
    }

    method reset {} {
	set $this-execmode reset
	$this-c needexecute
    }

    method locationclip {} {
	set $this-execmode location
	$this-c needexecute
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.location -relief groove -borderwidth 2
	frame $w.execmode -relief groove -borderwidth 2
	frame $w.whenexecute
	frame $w.executes -relief groove -borderwidth 2

	label $w.location.label -text "Location To Test"
        Tooltip $w.location "The clip test will be performed at this location to determine which elements are preserved."
	radiobutton $w.location.cell -text "Element Center" -variable $this-clip-location -value cell -command "$this locationclip"
	radiobutton $w.location.nodeone -text "One Node" -variable $this-clip-location -value nodeone -command "$this locationclip"
	radiobutton $w.location.nodeall -text "All Nodes" -variable $this-clip-location -value nodeall -command "$this locationclip"

	pack $w.location.label -side top -expand yes -fill both
	pack $w.location.cell $w.location.nodeone $w.location.nodeall -side top -anchor w

	label $w.execmode.label -text "Execute Action"
	radiobutton $w.execmode.replace -text "Replace" -variable $this-clipmode -value replace
	radiobutton $w.execmode.union -text "Union" -variable $this-clipmode -value union
	radiobutton $w.execmode.intersect -text "Intersect" -variable $this-clipmode -value intersect
	radiobutton $w.execmode.remove -text "Remove" -variable $this-clipmode -value remove

	pack $w.execmode.label -side top -fill both
	pack $w.execmode.replace $w.execmode.union $w.execmode.intersect $w.execmode.remove -side top -anchor w 

	checkbutton $w.whenexecute.check -text "Execute automatically" -variable $this-autoexecute

	checkbutton $w.whenexecute.icheck -text "Invert automatically" -variable $this-autoinvert -command "$this locationclip"
	
	pack $w.whenexecute.check $w.whenexecute.icheck -side top -anchor w -padx 10

	button $w.executes.invert -text "Invert" -command "$this invert"
	button $w.executes.undo -text "Undo" -command "$this undo"
	button $w.executes.reset -text "Reset" -command "$this reset"

	pack   $w.executes.invert $w.executes.undo $w.executes.reset -side left -e y -f both -padx 5 -pady 5

	pack $w.location $w.execmode $w.whenexecute $w.executes -side top -e y -f both -padx 5 -pady 5

	# Remove the (default) execute button so we can create our own
	# with the specific execrunmode command.
	makeSciButtonPanel $w $w $this -no_execute "\"Execute\" \"$this execrunmode\" \"\""
	moveToCursor $w
    }
}


