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


itcl_class SCIRun_Bundle_BundleGetPath {
    inherit Module
    constructor {config} {
        set name BundleGetPath
        set_defaults
    }

    method set_defaults {} {

        global $this-path1-name
        global $this-path2-name
        global $this-path3-name
        global $this-path1-listbox
        global $this-path2-listbox
        global $this-path3-listbox
        global $this-path-selection
        global $this-path1-entry
        global $this-path2-entry
        global $this-path3-entry


        set $this-path1-name "path1"
        set $this-path2-name "path2"
        set $this-path3-name "path3"
        set $this-path-selection ""

        set $this-path1-listbox ""
        set $this-path2-listbox ""
        set $this-path3-listbox ""
        set $this-path1-entry ""
        set $this-path2-entry ""
        set $this-path3-entry ""


    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-path1-name
        global $this-path2-name
        global $this-path3-name
        global $this-path-selection
        global $this-path1-listbox
        global $this-path2-listbox
        global $this-path3-listbox
        global $this-path1-entry
        global $this-path2-entry
        global $this-path3-entry

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "PATH OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "Path1"
        $childframe.pw add -label "Path2" 
        $childframe.pw add -label "Path3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set path1 [$childframe.pw childsite 0]
        set path2 [$childframe.pw childsite 1]
        set path3 [$childframe.pw childsite 2]

        frame $path1.name
        frame $path1.sel
        pack $path1.name -side top -fill x -expand yes -padx 5p
        pack $path1.sel -side top -fill both -expand yes -padx 5p

        label $path1.name.label -text "Name"
        entry $path1.name.entry -textvariable $this-path1-name
        set $this-path1-entry $path1.name.entry
        pack $path1.name.label -side left 
        pack $path1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $path1.sel.listbox  -selectioncommand [format "%s ChoosePath1" $this] -height 200p
        set $this-path1-listbox $path1.sel.listbox
        $path1.sel.listbox component listbox configure -listvariable $this-path-selection -selectmode browse
        pack $path1.sel.listbox -fill both -expand yes

        frame $path2.name
        frame $path2.sel
        pack $path2.name -side top -fill x -expand yes -padx 5p
        pack $path2.sel -side top -fill both -expand yes -padx 5p

        label $path2.name.label -text "Name"
        entry $path2.name.entry -textvariable $this-path2-name
        set $this-path2-entry $path2.name.entry
        pack $path2.name.label -side left 
        pack $path2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $path2.sel.listbox  -selectioncommand [format "%s ChoosePath2" $this] -height 200p
        set $this-path2-listbox $path2.sel.listbox
        $path2.sel.listbox component listbox configure -listvariable $this-path-selection -selectmode browse
        pack $path2.sel.listbox -fill both -expand yes
        
        frame $path3.name
        frame $path3.sel
        pack $path3.name -side top -fill x -expand yes -padx 5p
        pack $path3.sel -side top -fill both -expand yes -padx 5p

        label $path3.name.label -text "Name"
        entry $path3.name.entry -textvariable $this-path3-name
        set $this-path3-entry $path3.name.entry
        pack $path3.name.label -side left 
        pack $path3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $path3.sel.listbox  -selectioncommand [format "%s ChoosePath3" $this] -height 200p
        set $this-path3-listbox $path3.sel.listbox
        $path3.sel.listbox component listbox configure -listvariable $this-path-selection -selectmode browse
        pack $path3.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChoosePath1 { } {
        global $this-path1-listbox
        global $this-path1-name
        global $this-path-selection
        
        set pathnum [[set $this-path1-listbox] curselection]
        if [expr [string equal $pathnum ""] == 0] {
            set $this-path1-name  [lindex [set $this-path-selection] $pathnum] 
        }

    }

    method ChoosePath2 { } {
        global $this-path2-listbox
        global $this-path2-name
        global $this-path-selection
        
        set pathnum [[set $this-path2-listbox] curselection]
        if [expr [string equal $pathnum ""] == 0] {
            set $this-path2-name  [lindex [set $this-path-selection] $pathnum] 
        }
    }

    method ChoosePath3 { } {
        global $this-path3-listbox
        global $this-path3-name
        global $this-path-selection
        
        set pathnum [[set $this-path3-listbox] curselection]
        if [expr [string equal $pathnum ""] == 0] {
            set $this-path3-name  [lindex [set $this-path-selection] $pathnum] 
        }
    }
    
}
