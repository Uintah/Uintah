##
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


itcl_class ModelCreation_Math_ResizeMatrix {
    inherit Module
    constructor {config} {
        set name ResizeMatrix
        set_defaults
    }

    method set_defaults {} {
        global $this-dim-m
        global $this-dim-n
        
        set $this-dim-m 1
        set $this-dim-n 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        
        iwidgets::labeledframe $w.frame -labeltext "NEW MATRIX DIMENSIONS"
        set d [$w.frame childsite]
        pack $w.frame -fill both -expand yes
        
        label $d.lab1 -text "Number of Rows"
        entry $d.e1 -textvariable $this-dim-m        
        label $d.lab2 -text "Number of Columns"
        entry $d.e2 -textvariable $this-dim-n        
        
        grid $d.lab1 -row 0 -column 0  -sticky news
        grid $d.e1 -row 0 -column 1  -sticky news
        grid $d.lab2 -row 1 -column 0  -sticky news
        grid $d.e2 -row 1 -column 1  -sticky news

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


