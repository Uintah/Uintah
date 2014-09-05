#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2005 Scientific Computing and Imaging Institute,
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


catch {rename FlowVis2D ""}

itcl_class SCIRun_Visualization_FlowVis2D {
    inherit Module

    constructor {config} {
	set name FlowVis2D
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	
	set n "$this-c needexecute"
	set s "$this state"

        radiobutton $w.f.lic -text LIC -value 0 \
            -variable $this-vis_type
        radiobutton $w.f.ibfv -text IBFV -value 1 \
            -variable $this-vis_type
        radiobutton $w.f.lea -text LEA -value 2 \
            -variable $this-vis_type
        label $w.f.advl -text "advections per frame"
        scale $w.f.adv -variable $this-adv_accums -from 0 -to 20
        label $w.f.convl -text "convolutions per frame"
        scale $w.f.conv -variable $this-conv_accums -from 0 -to 20
        button $w.f.clear -text "Clear buffers" \
            -command "$this clear"
    
        pack  $w.f.lic $w.f.ibfv $w.f.lea $w.f.clear \
            $w.f.advl $w.f.adv $w.f.convl $w.f.conv -side top

	makeSciButtonPanel $w $w $this
	moveToCursor $w

        bind $w.f.adv <ButtonRelease> $n       
        bind $w.f.conv <ButtonRelease> $n
    }
    method clear {} {
        set $this-clear 1
        $this-c needexecute
    }
}