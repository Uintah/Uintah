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


itcl_class SCIRun_Bundle_BundleGetBundle {
    inherit Module
    constructor {config} {
        set name BundleGetBundle
        set_defaults
    }

    method set_defaults {} {

        global $this-bundle1-name
        global $this-bundle2-name
        global $this-bundle3-name
        global $this-bundle1-listbox
        global $this-bundle2-listbox
        global $this-bundle3-listbox
        global $this-bundle-selection
        global $this-bundle1-entry
        global $this-bundle2-entry
        global $this-bundle3-entry


        set $this-bundle1-name "bundle1"
        set $this-bundle2-name "bundle2"
        set $this-bundle3-name "bundle3"
        set $this-bundle-selection ""

        set $this-bundle1-listbox ""
        set $this-bundle2-listbox ""
        set $this-bundle3-listbox ""
        set $this-bundle1-entry ""
        set $this-bundle2-entry ""
        set $this-bundle3-entry ""


    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input bundle names

        global $this-bundle1-name
        global $this-bundle2-name
        global $this-bundle3-name
        global $this-bundle-selection
        global $this-bundle1-listbox
        global $this-bundle2-listbox
        global $this-bundle3-listbox
        global $this-bundle1-entry
        global $this-bundle2-entry
        global $this-bundle3-entry

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "BUNDLE SUB-BUNDLE OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "Bundle1"
        $childframe.pw add -label "Bundle2" 
        $childframe.pw add -label "Bundle3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set bundle1 [$childframe.pw childsite 0]
        set bundle2 [$childframe.pw childsite 1]
        set bundle3 [$childframe.pw childsite 2]

        frame $bundle1.name
        frame $bundle1.sel
        pack $bundle1.name -side top -fill x -expand yes -padx 5p
        pack $bundle1.sel -side top -fill both -expand yes -padx 5p

        label $bundle1.name.label -text "Name"
        entry $bundle1.name.entry -textvariable $this-bundle1-name
        set $this-bundle1-entry $bundle1.name.entry
        pack $bundle1.name.label -side left 
        pack $bundle1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $bundle1.sel.listbox  -selectioncommand [format "%s ChooseBundle1" $this] -height 200p
        set $this-bundle1-listbox $bundle1.sel.listbox
        $bundle1.sel.listbox component listbox configure -listvariable $this-bundle-selection -selectmode browse
        pack $bundle1.sel.listbox -fill both -expand yes

        frame $bundle2.name
        frame $bundle2.sel
        pack $bundle2.name -side top -fill x -expand yes -padx 5p
        pack $bundle2.sel -side top -fill both -expand yes -padx 5p

        label $bundle2.name.label -text "Name"
        entry $bundle2.name.entry -textvariable $this-bundle2-name
        set $this-bundle2-entry $bundle2.name.entry
        pack $bundle2.name.label -side left 
        pack $bundle2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $bundle2.sel.listbox  -selectioncommand [format "%s ChooseBundle2" $this] -height 200p
        set $this-bundle2-listbox $bundle2.sel.listbox
        $bundle2.sel.listbox component listbox configure -listvariable $this-bundle-selection -selectmode browse
        pack $bundle2.sel.listbox -fill both -expand yes
        
        frame $bundle3.name
        frame $bundle3.sel
        pack $bundle3.name -side top -fill x -expand yes -padx 5p
        pack $bundle3.sel -side top -fill both -expand yes -padx 5p

        label $bundle3.name.label -text "Name"
        entry $bundle3.name.entry -textvariable $this-bundle3-name
        set $this-bundle3-entry $bundle3.name.entry
        pack $bundle3.name.label -side left 
        pack $bundle3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $bundle3.sel.listbox  -selectioncommand [format "%s ChooseBundle3" $this] -height 200p
        set $this-bundle3-listbox $bundle3.sel.listbox
        $bundle3.sel.listbox component listbox configure -listvariable $this-bundle-selection -selectmode browse
        pack $bundle3.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChooseBundle1 { } {
        global $this-bundle1-listbox
        global $this-bundle1-name
        global $this-bundle-selection
        
        set bundlenum [[set $this-bundle1-listbox] curselection]
        if [expr [string equal $bundlenum ""] == 0] {
            set $this-bundle1-name  [lindex [set $this-bundle-selection] $bundlenum] 
        }

    }

    method ChooseBundle2 { } {
        global $this-bundle2-listbox
        global $this-bundle2-name
        global $this-bundle-selection
        
        set bundlenum [[set $this-bundle2-listbox] curselection]
        if [expr [string equal $bundlenum ""] == 0] {
            set $this-bundle2-name  [lindex [set $this-bundle-selection] $bundlenum] 
        }
    }

    method ChooseBundle3 { } {
        global $this-bundle3-listbox
        global $this-bundle3-name
        global $this-bundle-selection
        
        set bundlenum [[set $this-bundle3-listbox] curselection]
        if [expr [string equal $bundlenum ""] == 0] {
            set $this-bundle3-name  [lindex [set $this-bundle-selection] $bundlenum] 
        }
    }
    
}
