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



catch {rename BioPSE_Forward_ApplyFEMCurrentSource ""}

itcl_class BioPSE_Forward_ApplyFEMCurrentSource {
    inherit Module
    constructor {config} {
	set name ApplyFEMCurrentSource
	set_defaults
    }
    method set_defaults {} {
	global $this-sourceNodeTCL
	global $this-sinkNodeTCL
	global $this-modeTCL
	set $this-sourceNodeTCL 0
	set $this-sinkNodeTCL 1
	set $this-modeTCL dipole
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
	global $this-sourceNodeTCL
	global $this-sinkNodeTCL
	global $this-modeTCL

	set w .ui[modname]
	if {[winfo exists $w]} {
	    return
	}
	toplevel $w
	
	make_labeled_radio $w.mode "Source model:" "" left $this-modeTCL \
	    {dipole {"sources and sinks"}}
	make_entry $w.source "Source electrode:" $this-sourceNodeTCL "$this-c needexecute"
	make_entry $w.sink "Sink electrode:" $this-sinkNodeTCL "$this-c needexecute"
	bind $w.source <Return> "$this-c needexecute"
	bind $w.sink <Return> "$this-c needexecute"
	pack $w.mode $w.source $w.sink -side top -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}
