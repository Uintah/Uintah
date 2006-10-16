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


itcl_class SCIRun_FieldsGeometry_ChangeFieldBounds {
    inherit Module

    constructor {config} {
        set name ChangeFieldBounds

	# The width of the first column of the data display.
	setGlobal $this-firstwidth 12
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
