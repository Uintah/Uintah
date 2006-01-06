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


itcl_class SCIRun_Bundle_BundleSetString {
    inherit Module
    constructor {config} {
        set name BundleSetString
        set_defaults
    }

    method set_defaults {} {

        global $this-string1-name
        global $this-string2-name
        global $this-string3-name
        global $this-bundlename
        

        set $this-string1-name "string1"
        set $this-string2-name "string2"
        set $this-string3-name "string3"
        set $this-bundlename ""

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
        global $this-bundlename

        toplevel $w 

        wm minsize $w 100 150

        
        iwidgets::labeledframe $w.frame -labeltext "STRING INPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x
        frame $w.frame2
        pack $w.frame2 -fill x
        label $w.frame2.label -text "Name bundle object :"
        entry $w.frame2.entry -textvariable $this-bundlename
        pack $w.frame2.label -side left 
        pack $w.frame2.entry -side left -fill x

        iwidgets::tabnotebook $childframe.pw -height 100 -tabpos n
        $childframe.pw add -label "String1"
        $childframe.pw add -label "String2" 
        $childframe.pw add -label "String3" 
        $childframe.pw select 0

        pack $childframe.pw -fill x -expand yes

        set string1 [$childframe.pw childsite 0]
        set string2 [$childframe.pw childsite 1]
        set string3 [$childframe.pw childsite 2]

        frame $string1.name
        frame $string1.options
        pack $string1.name $string1.options -side top -fill x -expand yes -padx 5p

        label $string1.name.label -text "Name"
        entry $string1.name.entry -textvariable $this-string1-name
        pack $string1.name.label -side left 
        pack $string1.name.entry -side left -fill x -expand yes
        
        frame $string2.name
        frame $string2.options
        pack $string2.name $string2.options -side top -fill x -expand yes -padx 5p

        label $string2.name.label -text "Name"
        entry $string2.name.entry -textvariable $this-string2-name
        pack $string2.name.label -side left 
        pack $string2.name.entry -side left -fill x -expand yes
        
        frame $string3.name
        frame $string3.options
        pack $string3.name $string3.options -side top -fill x -expand yes -padx 5p

        label $string3.name.label -text "Name"
        entry $string3.name.entry -textvariable $this-string3-name
        pack $string3.name.label -side left 
        pack $string3.name.entry -side left -fill x -expand yes


        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
