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


itcl_class SCIRun_Bundle_BundleSetColorMap {
    inherit Module
    constructor {config} {
        set name BundleSetColorMap
        set_defaults
    }

    method set_defaults {} {

        global $this-colormap1-name
        global $this-colormap2-name
        global $this-colormap3-name
        global $this-bundlename
        
        set $this-colormap1-name "colormap1"
        set $this-colormap2-name "colormap2"
        set $this-colormap3-name "colormap3"
        set $this-bundlename ""

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
        global $this-bundlename

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "COLORMAP INPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x
        frame $w.frame2
        pack $w.frame2 -fill x
        label $w.frame2.label -text "Name bundle object :"
        entry $w.frame2.entry -textvariable $this-bundlename
        pack $w.frame2.label -side left 
        pack $w.frame2.entry -side left -fill x

        iwidgets::tabnotebook $childframe.pw -height 100 -tabpos n
        $childframe.pw add -label "ColorMap1"
        $childframe.pw add -label "ColorMap2" 
        $childframe.pw add -label "ColorMap3" 
        $childframe.pw select 0

        pack $childframe.pw -fill x -expand yes

        set colormap1 [$childframe.pw childsite 0]
        set colormap2 [$childframe.pw childsite 1]
        set colormap3 [$childframe.pw childsite 2]

        frame $colormap1.name
        frame $colormap1.options
        pack $colormap1.name $colormap1.options -side top -fill x -expand yes -padx 5p

        label $colormap1.name.label -text "Name"
        entry $colormap1.name.entry -textvariable $this-colormap1-name
        pack $colormap1.name.label -side left 
        pack $colormap1.name.entry -side left -fill x -expand yes
        
        frame $colormap2.name
        frame $colormap2.options
        pack $colormap2.name $colormap2.options -side top -fill x -expand yes -padx 5p

        label $colormap2.name.label -text "Name"
        entry $colormap2.name.entry -textvariable $this-colormap2-name
        pack $colormap2.name.label -side left 
        pack $colormap2.name.entry -side left -fill x -expand yes
        
        frame $colormap3.name
        frame $colormap3.options
        pack $colormap3.name $colormap3.options -side top -fill x -expand yes -padx 5p

        label $colormap3.name.label -text "Name"
        entry $colormap3.name.entry -textvariable $this-colormap3-name
        pack $colormap3.name.label -side left 
        pack $colormap3.name.entry -side left -fill x -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
