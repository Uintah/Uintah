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

 #  MatrixSend.tcl: Send matrix to a host:port
 #  Written by:
 #   Oleg
 #   Department of Computer Science
 #   University of Utah
 #   01Jan05
 ##

package require Iwidgets 3.0

catch {rename MatlabInterface_DataIO_Matlab ""}

itcl_class MatlabInterface_DataIO_Matlab {
    inherit Module
    constructor {config} {
        set name Matlab
        set_defaults
    }
    method set_defaults {} {
        global $this-cmdTCL
	global $this-hpTCL
	set $this-cmdTCL ""
	set $this-hpTCL "127.0.0.1:5517"
    }

    method update_text {} {
	set w .ui[modname]
	set $this-cmdTCL [$w.f.cmd.st get 1.0 end]
    }

    method send_text {} {
	$this update_text
	$this-c needexecute
    }

    method ui {} {
        set n "$this-c needexecute "
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

	global $this-cmdTCL
        global $this-hpTCL
	
        toplevel $w
        wm minsize $w 100 30
        
        frame $w.f
        pack $w.f -side top -fill both -expand yes
	
        frame $w.f.hp
        frame $w.f.cmd
	pack $w.f.hp $w.f.cmd -side top -fill both -expand yes

	label $w.f.hp.l -text "host:port : "
	entry $w.f.hp.e -relief sunken -width 21 -textvariable $this-hpTCL
	pack $w.f.hp.l -side left -padx 5 -pady 5
	pack $w.f.hp.e -side left -fill x -expand yes -padx 5 -pady 5

	option add *textBackground white	
	iwidgets::scrolledtext $w.f.cmd.st -vscrollmode dynamic \
		-labeltext "Matlab Commands"
	set cmmd "$this send_text"

	bind $w.f.cmd.st <Leave> "$this update_text"
	$w.f.cmd.st insert end [set $this-cmdTCL]

	button $w.f.cmd.execute -text "Execute" -command "$this-c ForceExecute"
	pack $w.f.cmd.st -padx 10 -pady 10 -fill both -expand yes
	pack $w.f.cmd.execute -side top
    }
}
