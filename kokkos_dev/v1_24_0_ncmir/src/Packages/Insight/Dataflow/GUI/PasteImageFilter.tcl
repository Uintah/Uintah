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

#    File   : PasteImageFilter.tcl
#    Author : Darby Van Uitert
#    Date   : January 2004

itcl_class Insight_Filters_PasteImageFilter {
    inherit Module
    constructor {config} {
        set name PasteImageFilter
        set_defaults
    }

    method set_defaults {} {
	global $this-size0
	global $this-size1
	global $this-size2
	global $this-fill_value
	global $this-axis
	global $this-index

	set $this-size0 0
	set $this-size1 0
	set $this-size2 0
	set $this-fill_value 0
	set $this-axis 2
	set $this-index 0
    }


    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 80
        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes

	make_entry $w.size0 "Size X:" $this-size0 5
	make_entry $w.size1 "Size Y:" $this-size1 5
	make_entry $w.size2 "Size Z:" $this-size2 5
	make_entry $w.fill "Initialize Value:" $this-fill_value 5

	button $w.generate -text "Generate Volume" \
	    -command "$this-c generate_volume"

	make_entry $w.axis "Axis: " $this-axis 5
	make_entry $w.index "Index: " $this-index 5

	pack $w.size0 $w.size1 $w.size2 $w.fill \
	    $w.generate $w.axis $w.index -side top -anchor nw

        makeSciButtonPanel $w $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }

    method make_entry {w text v {wi -1}} {
        frame $w
        label $w.l -text "$text" 
        pack $w.l -side left
        global $v
	if {$wi == -1} {
	    entry $w.e -textvariable $v -width 5
	} else {
	    entry $w.e -textvariable $v -width $wi
	}
        pack $w.e -side right
    }
}


