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


itcl_class SCIRun_Bundle_BundleSetPath {
    inherit Module
    constructor {config} {
        set name BundleSetPath
        set_defaults
    }

    method set_defaults {} {

        global $this-path1-name
        global $this-path2-name
        global $this-path3-name
        global $this-path1-usename
        global $this-path2-usename
        global $this-path3-usename
        global $this-bundlename
        

        set $this-path1-name "path1"
        set $this-path2-name "path2"
        set $this-path3-name "path3"
        set $this-path1-usename 0
        set $this-path2-usename 0
        set $this-path3-usename 0
        set $this-bundlename ""

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
        global $this-path1-usename
        global $this-path2-usename
        global $this-path3-usename
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
        $childframe.pw add -label "Path1"
        $childframe.pw add -label "Path2" 
        $childframe.pw add -label "Path3" 
        $childframe.pw select 0

        pack $childframe.pw -fill x -expand yes

        set path1 [$childframe.pw childsite 0]
        set path2 [$childframe.pw childsite 1]
        set path3 [$childframe.pw childsite 2]

        frame $path1.name
        frame $path1.options
        pack $path1.name $path1.options -side top -fill x -expand yes -padx 5p

        label $path1.name.label -text "Name"
        entry $path1.name.entry -textvariable $this-path1-name
        pack $path1.name.label -side left 
        pack $path1.name.entry -side left -fill x -expand yes
        checkbutton $path1.options.usename -text "Use object name" -variable $this-path1-usename
        pack $path1.options.usename -side top -fill x
        
        frame $path2.name
        frame $path2.options
        pack $path2.name $path2.options -side top -fill x -expand yes -padx 5p

        label $path2.name.label -text "Name"
        entry $path2.name.entry -textvariable $this-path2-name
        pack $path2.name.label -side left 
        pack $path2.name.entry -side left -fill x -expand yes
        checkbutton $path2.options.usename -text "Use object name" -variable $this-path2-usename
        pack $path2.options.usename -side top -fill x
        
        frame $path3.name
        frame $path3.options
        pack $path3.name $path3.options -side top -fill x -expand yes -padx 5p

        label $path3.name.label -text "Name"
        entry $path3.name.entry -textvariable $this-path3-name
        pack $path3.name.label -side left 
        pack $path3.name.entry -side left -fill x -expand yes
        checkbutton $path3.options.usename -text "Use object name" -variable $this-path3-usename
        pack $path3.options.usename -side top -fill x

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
