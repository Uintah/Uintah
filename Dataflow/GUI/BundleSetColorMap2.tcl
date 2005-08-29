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


itcl_class SCIRun_Bundle_BundleSetColorMap2 {
    inherit Module
    constructor {config} {
        set name BundleSetColorMap2
        set_defaults
    }

    method set_defaults {} {

        global $this-colormap21-name
        global $this-colormap22-name
        global $this-colormap23-name
        global $this-bundlename
        

        set $this-colormap21-name "colormap21"
        set $this-colormap22-name "colormap22"
        set $this-colormap23-name "colormap23"
        set $this-bundlename ""

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
        global $this-bundlename

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "COLORMAP2 INPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x
        frame $w.frame2
        pack $w.frame2 -fill x
        label $w.frame2.label -text "Name bundle object :"
        entry $w.frame2.entry -textvariable $this-bundlename
        pack $w.frame2.label -side left 
        pack $w.frame2.entry -side left -fill x

        iwidgets::tabnotebook $childframe.pw -height 100 -tabpos n
        $childframe.pw add -label "ColorMap21"
        $childframe.pw add -label "ColorMap22" 
        $childframe.pw add -label "ColorMap23" 
        $childframe.pw select 0

        pack $childframe.pw -fill x -expand yes

        set colormap21 [$childframe.pw childsite 0]
        set colormap22 [$childframe.pw childsite 1]
        set colormap23 [$childframe.pw childsite 2]

        frame $colormap21.name
        frame $colormap21.options
        pack $colormap21.name $colormap21.options -side top -fill x -expand yes -padx 5p

        label $colormap21.name.label -text "Name"
        entry $colormap21.name.entry -textvariable $this-colormap21-name
        pack $colormap21.name.label -side left 
        pack $colormap21.name.entry -side left -fill x -expand yes
        
        frame $colormap22.name
        frame $colormap22.options
        pack $colormap22.name $colormap22.options -side top -fill x -expand yes -padx 5p

        label $colormap22.name.label -text "Name"
        entry $colormap22.name.entry -textvariable $this-colormap22-name
        pack $colormap22.name.label -side left 
        pack $colormap22.name.entry -side left -fill x -expand yes
        
        frame $colormap23.name
        frame $colormap23.options
        pack $colormap23.name $colormap23.options -side top -fill x -expand yes -padx 5p

        label $colormap23.name.label -text "Name"
        entry $colormap23.name.entry -textvariable $this-colormap23-name
        pack $colormap23.name.label -side left 
        pack $colormap23.name.entry -side left -fill x -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
