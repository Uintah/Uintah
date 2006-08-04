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



catch {rename BioPSE_NeuroFEM_InverseIPM ""}

itcl_class BioPSE_NeuroFEM_InverseIPM {
    inherit Module
    constructor {config} {
        set name InverseIPM
        set_defaults
    }

    method set_defaults {} {
	global $this-ipm_pathTCL
	global $this-numchanTCL
	global $this-numsamplesTCL
	global $this-startsampleTCL
	global $this-stopsampleTCL
	global $this-associativityTCL
	global $this-posxTCL
	global $this-posyTCL
	global $this-poszTCL
	global $this-dirxTCL
	global $this-diryTCL
	global $this-dirzTCL
	global $this-eps_matTCL
	set $this-ipm_pathTCL "ipm_linux_dbx"
	set $this-numchanTCL  71
	set $this-numsamplesTCL 1
	set $this-startsampleTCL 0
	set $this-stopsampleTCL 1
	set $this-associativityTCL 1
	set $this-posxTCL 0.097
	set $this-posyTCL 0.154
	set $this-poszTCL 0.128
	set $this-dirxTCL 1.0
	set $this-diryTCL 0.0
	set $this-dirzTCL 0.0
	set $this-eps_matTCL 1e-2
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
	global $this-ipm_pathTCL
	global $this-numchanTCL
	global $this-numsamplesTCL
	global $this-startsampleTCL
	global $this-stopsampleTCL
	global $this-associativityTCL
	global $this-posxTCL
	global $this-posyTCL
	global $this-poszTCL
	global $this-dirxTCL
	global $this-diryTCL
	global $this-dirzTCL
	global $this-eps_matTCL

        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	make_entry $w.ipm_path "Path to NeuroFEM execution file: " $this-ipm_pathTCL "$this-c needexecute"
	make_entry $w.numchan "Number of Channel: " $this-numchanTCL "$this-c needexecute"
	make_entry $w.numsamples "Number of Samples: " $this-numsamplesTCL "$this-c needexecute"
	make_entry $w.startsample "Start Sample: " $this-startsampleTCL "$this-c needexecute"
	make_entry $w.stopsample "Stop Sample: " $this-stopsampleTCL "$this-c needexecute"
	make_entry $w.associativity "Lead Field Basis(yes=1, no=0): " $this-associativityTCL "$this-c needexecute"
	make_entry $w.posx "Initial Guess Position X: " $this-posxTCL "$this-c needexecute"
	make_entry $w.posy "Initial Guess Position Y: " $this-posyTCL "$this-c needexecute"
	make_entry $w.posz "Initial Guess Position Z: " $this-poszTCL "$this-c needexecute"
	make_entry $w.dirx "Initial Guess Moment X: " $this-dirxTCL "$this-c needexecute"
	make_entry $w.diry "Initial Guess Moment Y: " $this-diryTCL "$this-c needexecute"
	make_entry $w.dirz "Initial Guess Moment Z: " $this-dirzTCL "$this-c needexecute"
	make_entry $w.eps_mat "Pebbles Solver EPS_MAT: " $this-eps_matTCL "$this-c needexecute"

	bind $w.numchan <Return> "$this-c needexecute"
	bind $w.numsamples <Return> "$this-c needexecute"
	bind $w.startsample <Return> "$this-c needexecute"
	bind $w.stopsample <Return> "$this-c needexecute"
	bind $w.associativity <Return> "$this-c needexecute"
	bind $w.posx <Return> "$this-c needexecute"
	bind $w.posy <Return> "$this-c needexecute"
	bind $w.posz <Return> "$this-c needexecute"
	bind $w.dirx <Return> "$this-c needexecute"
	bind $w.diry <Return> "$this-c needexecute"
	bind $w.dirz <Return> "$this-c needexecute"
	bind $w.eps_mat <Return> "$this-c needexecute"

	pack $w.ipm_path -side top -fill x
	pack $w.numchan $w.numsamples $w.startsample $w.stopsample $w.associativity -side top -fill x
	pack $w.posx $w.posy $w.posz $w.dirx $w.diry $w.dirz -side top -fill x
	pack $w.eps_mat -side top -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
}


