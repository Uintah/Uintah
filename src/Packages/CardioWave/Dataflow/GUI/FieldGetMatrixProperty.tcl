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


itcl_class CardioWave_Fields_FieldGetMatrixProperty {
    inherit Module
    constructor {config} {
        set name FieldGetMatrixProperty
        set_defaults
    }

    method set_defaults {} {

        global $this-matrix1-name
        global $this-matrix2-name
        global $this-matrix3-name
        global $this-matrix1-listbox
        global $this-matrix2-listbox
        global $this-matrix3-listbox
        global $this-matrix-selection
        global $this-matrix1-entry
        global $this-matrix2-entry
        global $this-matrix3-entry

        set $this-matrix1-name "matrix1"
        set $this-matrix2-name "matrix2"
        set $this-matrix3-name "matrix3"
        set $this-matrix-selection ""

        set $this-matrix1-listbox ""
        set $this-matrix2-listbox ""
        set $this-matrix3-listbox ""
        set $this-matrix1-entry ""
        set $this-matrix2-entry ""
        set $this-matrix3-entry ""
        set $this-transposenrrd1 0
        set $this-transposenrrd2 0
        set $this-transposenrrd3 0

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
        global $this-matrix-selection
        global $this-matrix1-listbox
        global $this-matrix2-listbox
        global $this-matrix3-listbox
        global $this-matrix1-entry
        global $this-matrix2-entry
        global $this-matrix3-entry
        
        toplevel $w 

        wm minsize $w 100 150
        
        iwidgets::labeledframe $w.frame -labeltext "MATRIX OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill both -expand yes

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "Matrix1"
        $childframe.pw add -label "Matrix2" 
        $childframe.pw add -label "Matrix3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set matrix1 [$childframe.pw childsite 0]
        set matrix2 [$childframe.pw childsite 1]
        set matrix3 [$childframe.pw childsite 2]

        frame $matrix1.name
        frame $matrix1.transpose
        frame $matrix1.sel
        pack $matrix1.name -side top -fill x -padx 5p
        pack $matrix1.transpose -side top -fill x -padx 5p
        pack $matrix1.sel -side top -fill both -expand yes -padx 5p

        label $matrix1.name.label -text "Name"
        entry $matrix1.name.entry -textvariable $this-matrix1-name
        set $this-matrix1-entry $matrix1.name.entry
        pack $matrix1.name.label -side left 
        pack $matrix1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $matrix1.sel.listbox  -selectioncommand [format "%s ChooseMatrix1" $this]
        set $this-matrix1-listbox $matrix1.sel.listbox
        $matrix1.sel.listbox component listbox configure -listvariable $this-matrix-selection -selectmode browse 
        pack $matrix1.sel.listbox -fill both -expand yes


        frame $matrix2.name
        frame $matrix2.sel
        frame $matrix2.transpose
        pack $matrix2.name -side top -fill x -padx 5p
        pack $matrix2.transpose -side top -fill x -padx 5p
        pack $matrix2.sel -side top -fill both -expand yes -padx 5p

        label $matrix2.name.label -text "Name"
        entry $matrix2.name.entry -textvariable $this-matrix2-name
        set $this-matrix2-entry $matrix2.name.entry
        pack $matrix2.name.label -side left 
        pack $matrix2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $matrix2.sel.listbox  -selectioncommand [format "%s ChooseMatrix2" $this]
        set $this-matrix2-listbox $matrix2.sel.listbox
        $matrix2.sel.listbox component listbox configure -listvariable $this-matrix-selection -selectmode browse 
        pack $matrix2.sel.listbox -fill both -expand yes

        frame $matrix3.name
        frame $matrix3.transpose
        frame $matrix3.sel
        pack $matrix3.name -side top -fill x -padx 5p
        pack $matrix3.transpose -side top -fill x -padx 5p
        pack $matrix3.sel -side top -fill both -expand yes -padx 5p

        label $matrix3.name.label -text "Name"
        entry $matrix3.name.entry -textvariable $this-matrix3-name
        set $this-matrix3-entry $matrix3.name.entry
        pack $matrix3.name.label -side left 
        pack $matrix3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $matrix3.sel.listbox  -selectioncommand [format "%s ChooseMatrix3" $this]
        set $this-matrix3-listbox $matrix3.sel.listbox
        $matrix3.sel.listbox component listbox configure -listvariable $this-matrix-selection -selectmode browse
        pack $matrix3.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChooseMatrix1 { } {
        global $this-matrix1-listbox
        global $this-matrix1-name
        global $this-matrix-selection
        
        set matrixnum [[set $this-matrix1-listbox] curselection]
        if [expr [string equal $matrixnum ""] == 0] {
            set $this-matrix1-name  [lindex [set $this-matrix-selection] $matrixnum] 
        }

    }

    method ChooseMatrix2 { } {
        global $this-matrix2-listbox
        global $this-matrix2-name
        global $this-matrix-selection
        
        set matrixnum [[set $this-matrix2-listbox] curselection]
        if [expr [string equal $matrixnum ""] == 0] {
            set $this-matrix2-name  [lindex [set $this-matrix-selection] $matrixnum] 
        }
    }

    method ChooseMatrix3 { } {
        global $this-matrix3-listbox
        global $this-matrix3-name
        global $this-matrix-selection
        
        set matrixnum [[set $this-matrix3-listbox] curselection]
        if [expr [string equal $matrixnum ""] == 0] {
            set $this-matrix3-name  [lindex [set $this-matrix-selection] $matrixnum] 
        }
    }
    
}
