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


catch {rename BioPSE_Inverse_DipoleSearch ""}

itcl_class BioPSE_Inverse_DipoleSearch {
    inherit Module

    constructor {config} {
	set name DipoleSearch
	set_defaults
    }

    method set_defaults {} {	
	global $this-use_cache_gui_
	set $this-use_cache_gui_ 1
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	
        toplevel $w
	wm minsize $w 100 50
	
	set n "$this-c needexecute "
        
	frame $w.g
        button $w.g.go -text "Execute" -relief raised -command $n 
        button $w.g.p -text "Pause" -relief raised -command "$this-c pause"
        button $w.g.np -text "Unpause" -relief raised -command "$this-c unpause"
	button $w.g.stop -text "Stop" -relief raised -command "$this-c stop"
	pack $w.g.go $w.g.p $w.g.np $w.g.stop -side left -fill x
	global $this-use_cache_gui_
	checkbutton $w.b -text "UseCache" -variable $this-use_cache_gui_
	pack $w.g $w.b -side top -fill x
    }
}

