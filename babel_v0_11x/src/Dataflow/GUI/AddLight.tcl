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

itcl_class SCIRun_Visualization_AddLight {
    inherit Module
    constructor {config} {
        set name AddLight
        set_defaults
# 	if { [set $this-on] == 1 } {
# 	    $this-c needexecute  
# 	}
    }

    method set_defaults {} {
	global $this-control_pos_saved
	global $this-control_x
	global $this-control_y
	global $this-control_z
	global $this-at_x
	global $this-at_y
	global $this-at_z
	global $this-type
	global $this-on
	set $this-type 0
	set $this-on 1
	set $this-control_pos_saved 0
	set $this-control_x 0
	set $this-control_y 0
	set $this-control_z 0
	set $this-at_x 0
	set $this-at_y 0
	set $this-at_z 1
	
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	
	set n "$this-c needexecute"

	make_labeled_radio $w.light_type "Light type:" $n left \
	    $this-type { {Directional 0} {Point 1} {Spot 2}}
	checkbutton  $w.on  -text "on/off" \
	    -variable $this-on \
	    -command $n
	pack $w.light_type $w.on  -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w
	
    }
}


