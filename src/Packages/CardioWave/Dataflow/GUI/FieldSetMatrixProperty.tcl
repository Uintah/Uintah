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


itcl_class CardioWave_Fields_FieldSetMatrixProperty {
    inherit Module
    constructor {config} {
        set name FieldSetMatrixProperty
        set_defaults
    }

    method set_defaults {} {

        global $this-matrix1-name
        global $this-matrix2-name
        global $this-matrix3-name
        global $this-matrix1-usename
        global $this-matrix2-usename
        global $this-matrix3-usename

        set $this-matrix1-name "matrix1"
        set $this-matrix2-name "matrix2"
        set $this-matrix3-name "matrix3"
        set $this-matrix1-usename 0
        set $this-matrix2-usename 0
        set $this-matrix3-usename 0

    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-matrix1-name
        global $this-matrix2-name
        global $this-matrix3-name
        global $this-matrix1-usename
        global $this-matrix2-usename
        global $this-matrix3-usename

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "NRRD INPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x
    
        iwidgets::tabnotebook $childframe.pw -height 100 -tabpos n
        $childframe.pw add -label "Matrix1"
        $childframe.pw add -label "Matrix2" 
        $childframe.pw add -label "Matrix3" 
        $childframe.pw select 0

        pack $childframe.pw -fill x -expand yes

        set matrix1 [$childframe.pw childsite 0]
        set matrix2 [$childframe.pw childsite 1]
        set matrix3 [$childframe.pw childsite 2]

        frame $matrix1.name
        frame $matrix1.options
        pack $matrix1.name $matrix1.options -side top -fill x -expand yes -padx 5p

        label $matrix1.name.label -text "Name"
        entry $matrix1.name.entry -textvariable $this-matrix1-name
        pack $matrix1.name.label -side left 
        pack $matrix1.name.entry -side left -fill x -expand yes
        checkbutton $matrix1.options.usename -text "Use object name" -variable $this-matrix1-usename
        pack $matrix1.options.usename -side top -fill x
        
        frame $matrix2.name
        frame $matrix2.options
        pack $matrix2.name $matrix2.options -side top -fill x -expand yes -padx 5p

        label $matrix2.name.label -text "Name"
        entry $matrix2.name.entry -textvariable $this-matrix2-name
        pack $matrix2.name.label -side left 
        pack $matrix2.name.entry -side left -fill x -expand yes
        checkbutton $matrix2.options.usename -text "Use object name" -variable $this-matrix2-usename
        pack $matrix2.options.usename -side top -fill x
        
        frame $matrix3.name
        frame $matrix3.options
        pack $matrix3.name $matrix3.options -side top -fill x -expand yes -padx 5p

        label $matrix3.name.label -text "Name"
        entry $matrix3.name.entry -textvariable $this-matrix3-name
        pack $matrix3.name.label -side left 
        pack $matrix3.name.entry -side left -fill x -expand yes
        checkbutton $matrix3.options.usename -text "Use object name" -variable $this-matrix3-usename
        pack $matrix3.options.usename -side top -fill x

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
