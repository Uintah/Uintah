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


itcl_class SCIRun_FieldsGeometry_ChangeFieldBounds {
    inherit Module
    constructor {config} {
        set name ChangeFieldBounds
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	global $this-box-scale
	set $this-box-scale -1.0

	global $this-resetting
	set $this-resetting 0

	# these won't be saved 
        global $this-inputcenterx
        global $this-inputcentery
        global $this-inputcenterz
        global $this-inputsizex
        global $this-inputsizey
        global $this-inputsizez
        set $this-inputcenterx "---"
        set $this-inputcentery "---"
        set $this-inputcenterz "---"
        set $this-inputsizex "---"
        set $this-inputsizey "---"
        set $this-inputsizez "---"

	# these will be saved
        global $this-outputcenterx
        global $this-outputcentery
        global $this-outputcenterz
	global $this-useoutputcenter
        global $this-outputsizex
        global $this-outputsizey
        global $this-outputsizez
	global $this-useoutputsize
        set $this-outputcenterx 0
        set $this-outputcentery 0
        set $this-outputcenterz 0
	set $this-useoutputcenter 0
        set $this-outputsizex 0
        set $this-outputsizey 0
        set $this-outputsizez 0
	set $this-useoutputsize 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	wm maxsize $w 438 292
	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Field Attributes" 
			       
	pack $w.att -side top -fill x -expand y
	set att [$w.att childsite]
	
        labelpair3 $att.l1 "Center (x,y,z)" \
	    $this-inputcenterx $this-inputcentery $this-inputcenterz
        labelpair3 $att.l2 "Size (x,y,z)" \
	    $this-inputsizex $this-inputsizey $this-inputsizez
	pack $att.l1 $att.l2 -side top -fill x

	iwidgets::Labeledframe $w.edit -labelpos nw \
		               -labeltext "Output Field Attributes" 
	pack $w.edit -side top
	set edit [$w.edit childsite]
	
        labelentry3 $edit.l1 "Center (x,y,z)" \
	    $this-outputcenterx $this-outputcentery $this-outputcenterz \
	    "$this-c needexecute" \
	    $this-useoutputcenter
        labelentry3 $edit.l2 "Size (x,y,z)" \
	    $this-outputsizex $this-outputsizey \
	    $this-outputsizez "$this-c needexecute" \
	    $this-useoutputsize

	pack $edit.l1 $edit.l2 -side top 

	makeSciButtonPanel $w $w $this \
	    "\"Reset Widget\" \"$this reset\" \"\"" \
	    "\"In to Out\" \"$this copy_attributes\" \"Copies the Input Field Attribute values\nto the Output Field Attribute text fields.\n(This is just for user convenience.)\" "
	moveToCursor $w	
    }

    method reset {} {
	global $this-resetting
	set $this-resetting 1
	$this-c needexecute
    }

    method labelpair3 { win text1 text2x text2y text2z } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ": " -width 2 -anchor w -just left
	label $win.l2x -textvar $text2x -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma1  -text ", " -anchor w -just left  \
	    -fore darkred -borderwidth 0
	label $win.l2y -textvar $text2y -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma2  -text ", " -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.l2z -textvar $text2z -anchor w -just left \
	    -fore darkred -borderwidth 0
	pack $win.l1 $win.colon \
	    $win.l2x $win.comma1 $win.l2y $win.comma2 $win.l2z \
	    -side left -padx 0
    } 

    method labelentry2 { win text1 text2 text3 var } {
	frame $win 
	pack $win -side top -padx 5
	global $var
	checkbutton $win.b -var $var
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	entry $win.l2 -width 10 -just left \
		-fore darkred -text $text2
	entry $win.l3 -width 10 -just left \
		-fore darkred -text $text3
	label $win.l4 -width 40
	pack $win.b $win.l1 $win.colon -side left
	pack $win.l2 $win.l3 $win.l4 -padx 5 -side left
    }

    method labelentry3 { win text1 text2 text3 text4 func var } {
	frame $win 
	pack $win -side top -padx 5
	global $var
	checkbutton $win.b -var $var
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	entry $win.l2 -width 10 -just left \
		-fore darkred -text $text2
	entry $win.l3 -width 10 -just left \
		-fore darkred -text $text3
	entry $win.l4 -width 10 -just left \
		-fore darkred -text $text4
	label $win.l5 -width 40
	pack $win.b $win.l1 $win.colon -side left
	pack $win.l2 $win.l3 $win.l4 $win.l5 -padx 5 -side left

	bind $win.l2 <Return> $func
	bind $win.l3 <Return> $func
	bind $win.l4 <Return> $func
    }

    method copy_attributes {} {
	set w .ui[modname]
	if {![winfo exists $w]} {
	    return
	}
	set att [$w.att childsite]
	set edit [$w.edit childsite]
	set $this-outputcenterx [set $this-inputcenterx]
	set $this-outputcentery [set $this-inputcentery]
	set $this-outputcenterz [set $this-inputcenterz]
	set $this-outputsizex [set $this-inputsizex]
	set $this-outputsizey [set $this-inputsizey]
	set $this-outputsizez [set $this-inputsizez]
    }
}
