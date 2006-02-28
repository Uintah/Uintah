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


itcl_class CardioWave_Math_ReduceBandWidth {
    inherit Module
    constructor {config} {
        set name ReduceBandWidth
        set_defaults
    }

    method set_defaults {} {

        global $this-method
        global $this-special
        global $this-input-bw
        global $this-output-bw

        set $this-method "rcm"
        set $this-special "negate"
        set $this-input-bw "Input Matrix BandWidth = ---"
        set $this-output-bw "Output Matrix BandWidth = ---"
    }

    method ui {} {
        
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        global $this-method
        global $this-special
        global $this-input-bw
        global $this-output-bw
        
        toplevel $w 

        iwidgets::labeledframe $w.mframe -labeltext "REDUCTION METHOD"
        set mframe [$w.mframe childsite]
        pack $w.mframe -fill x -expand yes

        radiobutton $mframe.cm -text "CutHill mcKee" -variable $this-method -value cm 
        radiobutton $mframe.rcm -text "Reverse CutHillmcKee" -variable $this-method -value rcm 
        pack $mframe.cm $mframe.rcm -fill x -expand yes -anchor w
        
        iwidgets::labeledframe $w.bframe -labeltext "BANDWIDTH"
        set bframe [$w.bframe childsite]
        pack $w.bframe -fill x -expand yes

        label $bframe.inputbw -textvariable $this-input-bw
        label $bframe.outputbw -textvariable $this-output-bw
        pack  $bframe.inputbw $bframe.outputbw -fill x -expand yes -anchor w

        
        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}
