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


itcl_class SCIRun_Bundle_BundleGetColorMap2 {
    inherit Module
    constructor {config} {
        set name BundleGetColorMap2
        set_defaults
    }

    method set_defaults {} {

        global $this-colormap21-name
        global $this-colormap22-name
        global $this-colormap23-name
        global $this-colormap21-listbox
        global $this-colormap22-listbox
        global $this-colormap23-listbox
        global $this-colormap2-selection
        global $this-colormap21-entry
        global $this-colormap22-entry
        global $this-colormap23-entry


        set $this-colormap21-name "colormap21"
        set $this-colormap22-name "colormap22"
        set $this-colormap23-name "colormap23"
        set $this-colormap2-selection ""

        set $this-colormap21-listbox ""
        set $this-colormap22-listbox ""
        set $this-colormap23-listbox ""
        set $this-colormap21-entry ""
        set $this-colormap22-entry ""
        set $this-colormap23-entry ""


    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-colormap21-name
        global $this-colormap22-name
        global $this-colormap23-name
        global $this-colormap2-selection
        global $this-colormap21-listbox
        global $this-colormap22-listbox
        global $this-colormap23-listbox
        global $this-colormap21-entry
        global $this-colormap22-entry
        global $this-colormap23-entry

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "COLORMAP2 OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "ColorMap2-1"
        $childframe.pw add -label "ColorMap2-2" 
        $childframe.pw add -label "ColorMap2-3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set colormap21 [$childframe.pw childsite 0]
        set colormap22 [$childframe.pw childsite 1]
        set colormap23 [$childframe.pw childsite 2]

        frame $colormap21.name
        frame $colormap21.sel
        pack $colormap21.name -side top -fill x -expand yes -padx 5p
        pack $colormap21.sel -side top -fill both -expand yes -padx 5p

        label $colormap21.name.label -text "Name"
        entry $colormap21.name.entry -textvariable $this-colormap21-name
        set $this-colormap21-entry $colormap21.name.entry
        pack $colormap21.name.label -side left 
        pack $colormap21.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $colormap21.sel.listbox  -selectioncommand [format "%s ChooseColorMap21" $this] -height 200p
        set $this-colormap21-listbox $colormap21.sel.listbox
        $colormap21.sel.listbox component listbox configure -listvariable $this-colormap2-selection -selectmode browse
        pack $colormap21.sel.listbox -fill both -expand yes

        frame $colormap22.name
        frame $colormap22.sel
        pack $colormap22.name -side top -fill x -expand yes -padx 5p
        pack $colormap22.sel -side top -fill both -expand yes -padx 5p

        label $colormap22.name.label -text "Name"
        entry $colormap22.name.entry -textvariable $this-colormap22-name
        set $this-colormap22-entry $colormap22.name.entry
        pack $colormap22.name.label -side left 
        pack $colormap22.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $colormap22.sel.listbox  -selectioncommand [format "%s ChooseColorMap22" $this] -height 200p
        set $this-colormap22-listbox $colormap22.sel.listbox
        $colormap22.sel.listbox component listbox configure -listvariable $this-colormap2-selection -selectmode browse
        pack $colormap22.sel.listbox -fill both -expand yes
        
        frame $colormap23.name
        frame $colormap23.sel
        pack $colormap23.name -side top -fill x -expand yes -padx 5p
        pack $colormap23.sel -side top -fill both -expand yes -padx 5p

        label $colormap23.name.label -text "Name"
        entry $colormap23.name.entry -textvariable $this-colormap23-name
        set $this-colormap23-entry $colormap23.name.entry
        pack $colormap23.name.label -side left 
        pack $colormap23.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $colormap23.sel.listbox  -selectioncommand [format "%s ChooseColorMap23" $this] -height 200p
        set $this-colormap23-listbox $colormap23.sel.listbox
        $colormap23.sel.listbox component listbox configure -listvariable $this-colormap2-selection -selectmode browse
        pack $colormap23.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChooseColorMap21 { } {
        global $this-colormap21-listbox
        global $this-colormap21-name
        global $this-colormap2-selection
        
        set colormap2num [[set $this-colormap21-listbox] curselection]
        if [expr [string equal $colormap2num ""] == 0] {
            set $this-colormap21-name  [lindex [set $this-colormap2-selection] $colormap2num] 
        }

    }

    method ChooseColorMap22 { } {
        global $this-colormap22-listbox
        global $this-colormap22-name
        global $this-colormap2-selection
        
        set colormap2num [[set $this-colormap22-listbox] curselection]
        if [expr [string equal $colormap2num ""] == 0] {
            set $this-colormap22-name  [lindex [set $this-colormap2-selection] $colormap2num] 
        }
    }

    method ChooseColorMap23 { } {
        global $this-colormap23-listbox
        global $this-colormap23-name
        global $this-colormap2-selection
        
        set colormap2num [[set $this-colormap23-listbox] curselection]
        if [expr [string equal $colormap2num ""] == 0] {
            set $this-colormap23-name  [lindex [set $this-colormap2-selection] $colormap2num] 
        }
    }
    
}
