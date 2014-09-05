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


itcl_class SCIRun_Bundle_BundleGetField {
    inherit Module
    constructor {config} {
        set name BundleGetField

        initGlobal $this-field1-listbox ""
        initGlobal $this-field2-listbox ""
        initGlobal $this-field3-listbox ""
        initGlobal $this-field1-entry ""
        initGlobal $this-field2-entry ""
        initGlobal $this-field3-entry ""
    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-field1-name
        global $this-field2-name
        global $this-field3-name
        global $this-field-selection
        global $this-field1-listbox
        global $this-field2-listbox
        global $this-field3-listbox
        global $this-field1-entry
        global $this-field2-entry
        global $this-field3-entry

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "BUNDLE FIELD OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "Field1"
        $childframe.pw add -label "Field2" 
        $childframe.pw add -label "Field3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set field1 [$childframe.pw childsite 0]
        set field2 [$childframe.pw childsite 1]
        set field3 [$childframe.pw childsite 2]

        frame $field1.name
        frame $field1.sel
        pack $field1.name -side top -fill x -expand yes -padx 5p
        pack $field1.sel -side top -fill both -expand yes -padx 5p

        label $field1.name.label -text "Name"
        entry $field1.name.entry -textvariable $this-field1-name
        set $this-field1-entry $field1.name.entry
        pack $field1.name.label -side left 
        pack $field1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $field1.sel.listbox  -selectioncommand [format "%s ChooseField1" $this] -height 200p
        set $this-field1-listbox $field1.sel.listbox
        $field1.sel.listbox component listbox configure -listvariable $this-field-selection -selectmode browse
        pack $field1.sel.listbox -fill both -expand yes

        frame $field2.name
        frame $field2.sel
        pack $field2.name -side top -fill x -expand yes -padx 5p
        pack $field2.sel -side top -fill both -expand yes -padx 5p

        label $field2.name.label -text "Name"
        entry $field2.name.entry -textvariable $this-field2-name
        set $this-field2-entry $field2.name.entry
        pack $field2.name.label -side left 
        pack $field2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $field2.sel.listbox  -selectioncommand [format "%s ChooseField2" $this] -height 200p
        set $this-field2-listbox $field2.sel.listbox
        $field2.sel.listbox component listbox configure -listvariable $this-field-selection -selectmode browse
        pack $field2.sel.listbox -fill both -expand yes
        
        frame $field3.name
        frame $field3.sel
        pack $field3.name -side top -fill x -expand yes -padx 5p
        pack $field3.sel -side top -fill both -expand yes -padx 5p

        label $field3.name.label -text "Name"
        entry $field3.name.entry -textvariable $this-field3-name
        set $this-field3-entry $field3.name.entry
        pack $field3.name.label -side left 
        pack $field3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $field3.sel.listbox  -selectioncommand [format "%s ChooseField3" $this] -height 200p
        set $this-field3-listbox $field3.sel.listbox
        $field3.sel.listbox component listbox configure -listvariable $this-field-selection -selectmode browse
        pack $field3.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChooseField1 { } {
        global $this-field1-listbox
        global $this-field1-name
        global $this-field-selection
        
        set fieldnum [[set $this-field1-listbox] curselection]
        if [expr [string equal $fieldnum ""] == 0] {
            set $this-field1-name  [lindex [set $this-field-selection] $fieldnum] 
        }

    }

    method ChooseField2 { } {
        global $this-field2-listbox
        global $this-field2-name
        global $this-field-selection
        
        set fieldnum [[set $this-field2-listbox] curselection]
        if [expr [string equal $fieldnum ""] == 0] {
            set $this-field2-name  [lindex [set $this-field-selection] $fieldnum] 
        }
    }

    method ChooseField3 { } {
        global $this-field3-listbox
        global $this-field3-name
        global $this-field-selection
        
        set fieldnum [[set $this-field3-listbox] curselection]
        if [expr [string equal $fieldnum ""] == 0] {
            set $this-field3-name  [lindex [set $this-field-selection] $fieldnum] 
        }
    }
    
}
