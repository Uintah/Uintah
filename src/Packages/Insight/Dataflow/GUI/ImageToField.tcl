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


itcl_class Insight_Converters_ImageToField {
    inherit Module
    constructor {config} {
        set name ImageToField

	global $this-copy

        set_defaults
    }

    method set_defaults {} {
	set $this-copy 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
	    set child [lindex [winfo children $w] 0]
	    
	    # $w withdrawn by $child's procedures
	    raise $child
	    return;
        }
        toplevel $w

	radiobutton $w.a -text "Reference ITK Image Data" \
	    -variable $this-copy \
	    -value 0
	pack $w.a -anchor nw
	Tooltip $w.a "Select to reference the\nimage data directly."

	radiobutton $w.b -text "Copy Data" \
	    -variable $this-copy \
	    -value 1
	pack $w.b -anchor nw
	Tooltip $w.b "Select to copy the\nimage data to a Field"


	frame $w.buttons
	makeSciButtonPanel $w.buttons $w $this
	moveToCursor $w
	pack $w.buttons -side top -anchor n -padx 5 -pady 5
    }
}


