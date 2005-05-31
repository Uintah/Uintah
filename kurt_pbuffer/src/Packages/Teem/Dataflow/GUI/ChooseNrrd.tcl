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


itcl_class Teem_NrrdData_ChooseNrrd {
    inherit Module
    constructor {config} {
        set name ChooseNrrd

	global $this-port-index
	global $this-usefirstvalid

        set_defaults
    }

    method set_defaults {} {
	set $this-port-index 0
	set $this-usefirstvalid 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

	frame $w.c
	pack $w.c -side top -e y -f both -padx 5 -pady 5
	
	label $w.c.l -text "Select input port: "
	entry $w.c.e -textvariable $this-port-index
	bind $w.c.e <Return> "$this-c needexecute"
	pack $w.c.l $w.c.e -side left

	checkbutton $w.valid -text "Use first valid nrrd" \
	    -variable $this-usefirstvalid \
	    -command "$this toggle_usefirstvalid"
	pack $w.valid -side top -anchor nw -padx 3 -pady 3

	toggle_usefirstvalid

	Tooltip $w.valid "Module will iterate over ports\nand use the first one with a valid handle."

	TooltipMultiline $w.c.l \
            "Specify the input port that should be routed to the output port.\n" \
            "Index is 0 based (ie: the first port is index 0, the second port 1, etc.)"
	TooltipMultiline $w.c.e \
            "Specify the input port that should be routed to the output port.\n" \
            "Index is 0 based (ie: the first port is index 0, the second port 1, etc.)"

	makeSciButtonPanel $w $w $this

	moveToCursor $w
    }

    method toggle_usefirstvalid {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    # grey out port stuff if checked
	    if {[set $this-usefirstvalid]} {
		$w.c.l configure -foreground grey64
		$w.c.e configure -foreground grey64
	    } else {
		$w.c.l configure -foreground black
		$w.c.e configure -foreground black
	    }
	}
    }
}


