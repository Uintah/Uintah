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


itcl_class SCIRun_Bundle_BundleGetString {
    inherit Module
    constructor {config} {
        set name BundleGetString
        set_defaults
    }

    method set_defaults {} {

        global $this-string1-name
        global $this-string2-name
        global $this-string3-name
        global $this-string1-listbox
        global $this-string2-listbox
        global $this-string3-listbox
        global $this-string-selection
        global $this-string1-entry
        global $this-string2-entry
        global $this-string3-entry


        set $this-string1-name "string1"
        set $this-string2-name "string2"
        set $this-string3-name "string3"
        set $this-string-selection ""

        set $this-string1-listbox ""
        set $this-string2-listbox ""
        set $this-string3-listbox ""
        set $this-string1-entry ""
        set $this-string2-entry ""
        set $this-string3-entry ""


    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-string1-name
        global $this-string2-name
        global $this-string3-name
        global $this-string-selection
        global $this-string1-listbox
        global $this-string2-listbox
        global $this-string3-listbox
        global $this-string1-entry
        global $this-string2-entry
        global $this-string3-entry

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "STRING OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "String1"
        $childframe.pw add -label "String2" 
        $childframe.pw add -label "String3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set string1 [$childframe.pw childsite 0]
        set string2 [$childframe.pw childsite 1]
        set string3 [$childframe.pw childsite 2]

        frame $string1.name
        frame $string1.sel
        pack $string1.name -side top -fill x -expand yes -padx 5p
        pack $string1.sel -side top -fill both -expand yes -padx 5p

        label $string1.name.label -text "Name"
        entry $string1.name.entry -textvariable $this-string1-name
        set $this-string1-entry $string1.name.entry
        pack $string1.name.label -side left 
        pack $string1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $string1.sel.listbox  -selectioncommand [format "%s ChooseString1" $this] -height 200p
        set $this-string1-listbox $string1.sel.listbox
        $string1.sel.listbox component listbox configure -listvariable $this-string-selection -selectmode browse
        pack $string1.sel.listbox -fill both -expand yes

        frame $string2.name
        frame $string2.sel
        pack $string2.name -side top -fill x -expand yes -padx 5p
        pack $string2.sel -side top -fill both -expand yes -padx 5p

        label $string2.name.label -text "Name"
        entry $string2.name.entry -textvariable $this-string2-name
        set $this-string2-entry $string2.name.entry
        pack $string2.name.label -side left 
        pack $string2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $string2.sel.listbox  -selectioncommand [format "%s ChooseString2" $this] -height 200p
        set $this-string2-listbox $string2.sel.listbox
        $string2.sel.listbox component listbox configure -listvariable $this-string-selection -selectmode browse
        pack $string2.sel.listbox -fill both -expand yes
        
        frame $string3.name
        frame $string3.sel
        pack $string3.name -side top -fill x -expand yes -padx 5p
        pack $string3.sel -side top -fill both -expand yes -padx 5p

        label $string3.name.label -text "Name"
        entry $string3.name.entry -textvariable $this-string3-name
        set $this-string3-entry $string3.name.entry
        pack $string3.name.label -side left 
        pack $string3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $string3.sel.listbox  -selectioncommand [format "%s ChooseString3" $this] -height 200p
        set $this-string3-listbox $string3.sel.listbox
        $string3.sel.listbox component listbox configure -listvariable $this-string-selection -selectmode browse
        pack $string3.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChooseString1 { } {
        global $this-string1-listbox
        global $this-string1-name
        global $this-string-selection
        
        set stringnum [[set $this-string1-listbox] curselection]
        if [expr [string equal $stringnum ""] == 0] {
            set $this-string1-name  [lindex [set $this-string-selection] $stringnum] 
        }

    }

    method ChooseString2 { } {
        global $this-string2-listbox
        global $this-string2-name
        global $this-string-selection
        
        set stringnum [[set $this-string2-listbox] curselection]
        if [expr [string equal $stringnum ""] == 0] {
            set $this-string2-name  [lindex [set $this-string-selection] $stringnum] 
        }
    }

    method ChooseString3 { } {
        global $this-string3-listbox
        global $this-string3-name
        global $this-string-selection
        
        set stringnum [[set $this-string3-listbox] curselection]
        if [expr [string equal $stringnum ""] == 0] {
            set $this-string3-name  [lindex [set $this-string-selection] $stringnum] 
        }
    }
    
}
