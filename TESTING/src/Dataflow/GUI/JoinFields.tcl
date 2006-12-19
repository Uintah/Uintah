#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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


itcl_class SCIRun_NewField_JoinFields {
    inherit Module

    constructor {config} {
        set name JoinFields
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        checkbutton $w.fpc -text "Force PointCloudField as output" \
          -variable $this-force-pointcloud
        pack $w.fpc

        checkbutton $w.fnm -text "Merge duplicate nodes" \
          -variable $this-force-nodemerge
        pack $w.fnm

        checkbutton $w.fnm2 -text "Only merge nodes with same value" \
          -variable $this-matchval
        pack $w.fnm2

        checkbutton $w.fmo -text "Merge mesh only, do not assign values" \
          -variable $this-meshonly
        pack $w.fmo

        iwidgets::entryfield $w.prec \
          -labeltext "Tolerance (in distance) for merging nodes" \
          -textvariable $this-tolerance
        pack $w.prec -side top -expand yes -fill x

        makeSciButtonPanel $w $w $this
        moveToCursor $w
     }
}
