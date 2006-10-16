##
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


itcl_class ModelCreation_Script_ParameterListMatrix {
    inherit Module
    constructor {config} {
        set name ParameterListMatrix
        set_defaults
    }

    method set_defaults {} {
        global $this-matrix-name
        global $this-matrix-listbox
        global $this-matrix-selection
        global $this-matrix-entry

        set $this-matrix-name "matrix"
        set $this-matrix-selection ""
        set $this-matrix-listbox ""
        set $this-matrix-entry ""
    }

    
    method choose_field {} {
        global $this-matrix-name
        global $this-matrix-selection
        
        set w .ui[modname]
        if {[winfo exists $w]} {
          set matrixnum [$w.sel.listbox curselection]
          if [expr [string equal $matrixnum ""] == 0] {
            set $this-matrix-name  [lindex [set $this-matrix-selection] $matrixnum] 
          }
        }
    }    


    method ui {} {
        global $this-matrix-name
        global $this-matrix-listbox
        global $this-matrix-selection
        global $this-matrix-entry    
    
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        wm minsize $w 100 50        
  
        frame $w.name
        frame $w.sel

        pack $w.name -side top -fill x -padx 5p
        pack $w.sel -side top -fill both -expand yes -padx 5p
        
        label $w.name.label -text "Matrix Name"
        entry $w.name.entry -textvariable $this-matrix-name
        pack $w.name.label -side left 
        pack $w.name.entry -side left -fill x -expand yes

        iwidgets::scrolledlistbox $w.sel.listbox -selectioncommand "$this choose_field"
        $w.sel.listbox component listbox configure -listvariable $this-matrix-selection -selectmode browse 
        pack $w.sel.listbox -fill both -expand yes
               
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


