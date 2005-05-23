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


itcl_class SCIRun_Bundle_BundleGetColorMap {
    inherit Module
    constructor {config} {
        set name BundleGetColorMap
        set_defaults
    }

    method set_defaults {} {

        global $this-colormap1-name
        global $this-colormap2-name
        global $this-colormap3-name
        global $this-colormap1-listbox
        global $this-colormap2-listbox
        global $this-colormap3-listbox
        global $this-colormap-selection
        global $this-colormap1-entry
        global $this-colormap2-entry
        global $this-colormap3-entry


        set $this-colormap1-name "colormap1"
        set $this-colormap2-name "colormap2"
        set $this-colormap3-name "colormap3"
        set $this-colormap-selection ""

        set $this-colormap1-listbox ""
        set $this-colormap2-listbox ""
        set $this-colormap3-listbox ""
        set $this-colormap1-entry ""
        set $this-colormap2-entry ""
        set $this-colormap3-entry ""


    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-colormap1-name
        global $this-colormap2-name
        global $this-colormap3-name
        global $this-colormap-selection
        global $this-colormap1-listbox
        global $this-colormap2-listbox
        global $this-colormap3-listbox
        global $this-colormap1-entry
        global $this-colormap2-entry
        global $this-colormap3-entry

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "BUNDLE FIELD OUTPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x

        iwidgets::tabnotebook $childframe.pw -height 160 -tabpos n
        $childframe.pw add -label "ColorMap1"
        $childframe.pw add -label "ColorMap2" 
        $childframe.pw add -label "ColorMap3" 
        $childframe.pw select 0

        pack $childframe.pw -fill both -expand yes

        set colormap1 [$childframe.pw childsite 0]
        set colormap2 [$childframe.pw childsite 1]
        set colormap3 [$childframe.pw childsite 2]

        frame $colormap1.name
        frame $colormap1.sel
        pack $colormap1.name -side top -fill x -expand yes -padx 5p
        pack $colormap1.sel -side top -fill both -expand yes -padx 5p

        label $colormap1.name.label -text "Name"
        entry $colormap1.name.entry -textvariable $this-colormap1-name
        set $this-colormap1-entry $colormap1.name.entry
        pack $colormap1.name.label -side left 
        pack $colormap1.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $colormap1.sel.listbox  -selectioncommand [format "%s ChooseColorMap1" $this] -height 200p
        set $this-colormap1-listbox $colormap1.sel.listbox
        $colormap1.sel.listbox component listbox configure -listvariable $this-colormap-selection -selectmode browse
        pack $colormap1.sel.listbox -fill both -expand yes

        frame $colormap2.name
        frame $colormap2.sel
        pack $colormap2.name -side top -fill x -expand yes -padx 5p
        pack $colormap2.sel -side top -fill both -expand yes -padx 5p

        label $colormap2.name.label -text "Name"
        entry $colormap2.name.entry -textvariable $this-colormap2-name
        set $this-colormap2-entry $colormap2.name.entry
        pack $colormap2.name.label -side left 
        pack $colormap2.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $colormap2.sel.listbox  -selectioncommand [format "%s ChooseColorMap2" $this] -height 200p
        set $this-colormap2-listbox $colormap2.sel.listbox
        $colormap2.sel.listbox component listbox configure -listvariable $this-colormap-selection -selectmode browse
        pack $colormap2.sel.listbox -fill both -expand yes
        
        frame $colormap3.name
        frame $colormap3.sel
        pack $colormap3.name -side top -fill x -expand yes -padx 5p
        pack $colormap3.sel -side top -fill both -expand yes -padx 5p

        label $colormap3.name.label -text "Name"
        entry $colormap3.name.entry -textvariable $this-colormap3-name
        set $this-colormap3-entry $colormap3.name.entry
        pack $colormap3.name.label -side left 
        pack $colormap3.name.entry -side left -fill x -expand yes
        
        iwidgets::scrolledlistbox $colormap3.sel.listbox  -selectioncommand [format "%s ChooseColorMap3" $this] -height 200p
        set $this-colormap3-listbox $colormap3.sel.listbox
        $colormap3.sel.listbox component listbox configure -listvariable $this-colormap-selection -selectmode browse
        pack $colormap3.sel.listbox -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    
    method ChooseColorMap1 { } {
        global $this-colormap1-listbox
        global $this-colormap1-name
        global $this-colormap-selection
        
        set colormapnum [[set $this-colormap1-listbox] curselection]
        if [expr [string equal $colormapnum ""] == 0] {
            set $this-colormap1-name  [lindex [set $this-colormap-selection] $colormapnum] 
        }

    }

    method ChooseColorMap2 { } {
        global $this-colormap2-listbox
        global $this-colormap2-name
        global $this-colormap-selection
        
        set colormapnum [[set $this-colormap2-listbox] curselection]
        if [expr [string equal $colormapnum ""] == 0] {
            set $this-colormap2-name  [lindex [set $this-colormap-selection] $colormapnum] 
        }
    }

    method ChooseColorMap3 { } {
        global $this-colormap3-listbox
        global $this-colormap3-name
        global $this-colormap-selection
        
        set colormapnum [[set $this-colormap3-listbox] curselection]
        if [expr [string equal $colormapnum ""] == 0] {
            set $this-colormap3-name  [lindex [set $this-colormap-selection] $colormapnum] 
        }
    }
    
}
