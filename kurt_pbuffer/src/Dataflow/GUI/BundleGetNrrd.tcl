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


itcl_class SCIRun_Bundle_BundleGetNrrd {
    inherit Module
    constructor {config} {
        set name BundleGetNrrd
        set_defaults
    }

    method set_defaults {} {

        global $this-nrrd1-name
        global $this-nrrd2-name
        global $this-nrrd3-name
        global $this-nrrd1-listbox
        global $this-nrrd2-listbox
        global $this-nrrd3-listbox
        global $this-nrrd-selection
        global $this-nrrd1-entry
        global $this-nrrd2-entry
        global $this-nrrd3-entry
        global $this-transposenrrd1
        global $this-transposenrrd2
        global $this-transposenrrd3

        set $this-nrrd1-name "nrrd1"
        set $this-nrrd2-name "nrrd2"
        set $this-nrrd3-name "nrrd3"
        set $this-nrrd-selection ""

        set $this-nrrd1-listbox ""
        set $this-nrrd2-listbox ""
        set $this-nrrd3-listbox ""
        set $this-nrrd1-entry ""
        set $this-nrrd2-entry ""
        set $this-nrrd3-entry ""
        set $this-transposenrrd1 0
        set $this-transposenrrd2 0
        set $this-transposenrrd3 0

    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input nrrd names

        global $this-nrrd1-name
        global $this-nrrd2-name
        global $this-nrrd3-name
        global $this-nrrd-selection
        global $this-nrrd1-listbox
        global $this-nrrd2-listbox
        global $this-nrrd3-listbox
        global $this-nrrd1-entry
        global $this-nrrd2-entry
        global $this-nrrd3-entry
        global $this-transposenrrd1
        global $this-transposenrrd2
        global $this-transposenrrd3
        
        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "BUNDLE NRRD OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill both -expand yes

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "Nrrd1"
        $childframe.pw add -label "Nrrd2" 
        $childframe.pw add -label "Nrrd3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set nrrd1 [$childframe.pw childsite 0]
        set nrrd2 [$childframe.pw childsite 1]
        set nrrd3 [$childframe.pw childsite 2]

        frame $nrrd1.name
        frame $nrrd1.sel
        frame $nrrd1.transpose
        pack $nrrd1.name -side top -fill x -padx 5p
        pack $nrrd1.transpose -side top -fill x -padx 5p
        pack $nrrd1.sel -side top -fill both -expand yes -padx 5p

        label $nrrd1.name.label -text "Name"
        entry $nrrd1.name.entry -textvariable $this-nrrd1-name
        set $this-nrrd1-entry $nrrd1.name.entry
        pack $nrrd1.name.label -side left 
        pack $nrrd1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $nrrd1.sel.listbox  -selectioncommand [format "%s ChooseNrrd1" $this] 
        set $this-nrrd1-listbox $nrrd1.sel.listbox
        $nrrd1.sel.listbox component listbox configure -listvariable $this-nrrd-selection -selectmode browse
        pack $nrrd1.sel.listbox -fill both -expand yes
        checkbutton $nrrd1.transpose.cb -variable $this-transposenrrd1 -text "Assume matrix data is transposed"
        pack $nrrd1.transpose.cb -side left -fill x

        frame $nrrd2.name
        frame $nrrd2.sel
        frame $nrrd2.transpose    
        pack $nrrd2.name -side top -fill x -padx 5p
        pack $nrrd2.transpose -side top -fill x -padx 5p
        pack $nrrd2.sel -side top -fill both -expand yes -padx 5p

        label $nrrd2.name.label -text "Name"
        entry $nrrd2.name.entry -textvariable $this-nrrd2-name
        set $this-nrrd2-entry $nrrd2.name.entry
        pack $nrrd2.name.label -side left 
        pack $nrrd2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $nrrd2.sel.listbox  -selectioncommand [format "%s ChooseNrrd2" $this] 
        set $this-nrrd2-listbox $nrrd2.sel.listbox
        $nrrd2.sel.listbox component listbox configure -listvariable $this-nrrd-selection -selectmode browse
        pack $nrrd2.sel.listbox -fill both -expand yes
        checkbutton $nrrd2.transpose.cb -variable $this-transposenrrd1 -text "Assume matrix data is transposed"
        pack $nrrd2.transpose.cb -side left -fill x
        
        frame $nrrd3.name
        frame $nrrd3.sel
        frame $nrrd3.transpose
        pack $nrrd3.name -side top -fill x -padx 5p
        pack $nrrd3.transpose -side top -fill x -padx 5p
        pack $nrrd3.sel -side top -fill both -expand yes -padx 5p

        label $nrrd3.name.label -text "Name"
        entry $nrrd3.name.entry -textvariable $this-nrrd3-name
        set $this-nrrd3-entry $nrrd3.name.entry
        pack $nrrd3.name.label -side left 
        pack $nrrd3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $nrrd3.sel.listbox  -selectioncommand [format "%s ChooseNrrd3" $this] 
        set $this-nrrd3-listbox $nrrd3.sel.listbox
        $nrrd3.sel.listbox component listbox configure -listvariable $this-nrrd-selection -selectmode browse
        pack $nrrd3.sel.listbox -fill both -expand yes
        checkbutton $nrrd3.transpose.cb -variable $this-transposenrrd1 -text "Assume matrix data is transposed"
        pack $nrrd3.transpose.cb -side left -fill x

        makeSciButtonPanel $w $w $this

    }
    
    
    method ChooseNrrd1 { } {
        global $this-nrrd1-listbox
        global $this-nrrd1-name
        global $this-nrrd-selection
        
        set nrrdnum [[set $this-nrrd1-listbox] curselection]
        if [expr [string equal $nrrdnum ""] == 0] {
            set $this-nrrd1-name  [lindex [set $this-nrrd-selection] $nrrdnum] 
        }

    }

    method ChooseNrrd2 { } {
        global $this-nrrd2-listbox
        global $this-nrrd2-name
        global $this-nrrd-selection
        
        set nrrdnum [[set $this-nrrd2-listbox] curselection]
        if [expr [string equal $nrrdnum ""] == 0] {
            set $this-nrrd2-name  [lindex [set $this-nrrd-selection] $nrrdnum] 
        }
    }

    method ChooseNrrd3 { } {
        global $this-nrrd3-listbox
        global $this-nrrd3-name
        global $this-nrrd-selection
        
        set nrrdnum [[set $this-nrrd3-listbox] curselection]
        if [expr [string equal $nrrdnum ""] == 0] {
            set $this-nrrd3-name  [lindex [set $this-nrrd-selection] $nrrdnum] 
        }
    }
    
}
