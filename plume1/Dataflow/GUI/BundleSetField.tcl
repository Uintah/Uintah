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


itcl_class SCIRun_Bundle_BundleSetField {
    inherit Module
    constructor {config} {
        set name BundleSetField
        set_defaults
    }

    method set_defaults {} {

        global $this-field1-name
        global $this-field2-name
        global $this-field3-name
        global $this-bundlename
        
        set $this-field1-name "field1"
        set $this-field2-name "field2"
        set $this-field3-name "field3"
        set $this-bundlename ""

    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        # input matrix names

        global $this-field1-name
        global $this-field2-name
        global $this-field3-name
        global $this-bundlename

        toplevel $w 

        wm minsize $w 100 150
        
        iwidgets::labeledframe $w.frame -labeltext "FIELD INPUTS"
        set childframe [$w.frame childsite]
        pack $w.frame -fill x
        frame $w.frame2
        pack $w.frame2 -fill x
        label $w.frame2.label -text "Name bundle object :"
        entry $w.frame2.entry -textvariable $this-bundlename
        pack $w.frame2.label -side left 
        pack $w.frame2.entry -side left -fill x

        iwidgets::tabnotebook $childframe.pw -height 100 -tabpos n
        $childframe.pw add -label "Field1"
        $childframe.pw add -label "Field2" 
        $childframe.pw add -label "Field3" 
        $childframe.pw select 0

        pack $childframe.pw -fill x -expand yes

        set field1 [$childframe.pw childsite 0]
        set field2 [$childframe.pw childsite 1]
        set field3 [$childframe.pw childsite 2]

        frame $field1.name
        frame $field1.options
        pack $field1.name $field1.options -side top -fill x -expand yes -padx 5p

        label $field1.name.label -text "Name"
        entry $field1.name.entry -textvariable $this-field1-name
        pack $field1.name.label -side left 
        pack $field1.name.entry -side left -fill x -expand yes
        
        frame $field2.name
        frame $field2.options
        pack $field2.name $field2.options -side top -fill x -expand yes -padx 5p

        label $field2.name.label -text "Name"
        entry $field2.name.entry -textvariable $this-field2-name
        pack $field2.name.label -side left 
        pack $field2.name.entry -side left -fill x -expand yes
        
        frame $field3.name
        frame $field3.options
        pack $field3.name $field3.options -side top -fill x -expand yes -padx 5p

        label $field3.name.label -text "Name"
        entry $field3.name.entry -textvariable $this-field3-name
        pack $field3.name.label -side left 
        pack $field3.name.entry -side left -fill x -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
