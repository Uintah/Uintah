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
#    File   : PVSpaceInterp.tcl
#    Author : Martin Cole
#    Date   : Tue May 10 08:48:37 2005

catch {rename VS_Fields_PVSpaceInterp ""}

itcl_class VS_Fields_PVSpaceInterp {
    inherit Module
    constructor {config} {
        set name PVSpaceInterp
        set_defaults
    }
    method set_defaults {} {
        global $this-sample_rate
        set $this-sample_rate 100.0

        global $this-phase_index
        set $this-phase_index 6

	global $this-vol_index
	set $this-vol_index 2

	global $this-lvp_index
	set $this-lvp_index 1

	global $this-rvp_index
	set $this-rvp_index 3

    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	frame $w.f.options
	pack $w.f.options -side top -expand yes

        iwidgets::entryfield $w.f.options.sr -labeltext "HIP sample rate:" \
	    -textvariable $this-sample_rate
        pack $w.f.options.sr -side top -expand yes -fill x

        iwidgets::entryfield $w.f.options.phasei -labeltext "phase index:" \
	    -textvariable $this-phase_index
        pack $w.f.options.phasei -side top -expand yes -fill x
	
	iwidgets::entryfield $w.f.options.voli -labeltext "LVV index:" \
	    -textvariable $this-vol_index
        pack $w.f.options.voli -side top -expand yes -fill x
	
	iwidgets::entryfield $w.f.options.lvpi -labeltext "LVP index:" \
	    -textvariable $this-lvp_index
        pack $w.f.options.lvpi -side top -expand yes -fill x

	iwidgets::entryfield $w.f.options.rvpi -labeltext "RVP index:" \
	    -textvariable $this-rvp_index
        pack $w.f.options.rvpi -side top -expand yes -fill x
	
	makeSciButtonPanel $w.f $w $this
	moveToCursor $w

	pack $w.f -expand 1 -fill x
    }
}


