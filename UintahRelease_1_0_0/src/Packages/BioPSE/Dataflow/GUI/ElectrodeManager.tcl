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



catch {rename BioPSE_Forward_ElectrodeManager ""}

itcl_class BioPSE_Forward_ElectrodeManager {
    inherit Module
    constructor {config} {
        set name ElectrodeManager
        set_defaults
    }

    method set_defaults {} {
        global $this-modelTCL
	global $this-numElTCL
	global $this-lengthElTCL
	global $this-startNodeIdxTCL
        set $this-modelTCL 1
	set $this-numElTCL 32
	set $this-lengthElTCL 0.027
	set $this-startNodeIdxTCL 0
    }

    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }

    method ui {} {
        global $this-modelTCL
	global $this-numElTCL
	global $this-lengthElTCL
	global $this-startNodeIdxTCL

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	make_entry $w.model "0=Continuum, 1=Gap: " $this-modelTCL "$this-c needexecute"
	make_entry $w.numEl "Number of electrodes: " $this-numElTCL "$this-c needexecute"
	make_entry $w.lengthEl "Electrode length: " $this-lengthElTCL "$this-c needexecute"
	make_entry $w.startNodeIdx "Boundary node index for start of first electrode: " $this-startNodeIdxTCL "$this-c needexecute"
	bind $w.model <Return> "$this-c needexecute"
	bind $w.numEl <Return> "$this-c needexecute"
	bind $w.lengthEl <Return> "$this-c needexecute"
	bind $w.startNodeIdx <Return> "$this-c needexecute"
	pack $w.model $w.numEl $w.lengthEl $w.startNodeIdx -side top -fill x

    }
}


