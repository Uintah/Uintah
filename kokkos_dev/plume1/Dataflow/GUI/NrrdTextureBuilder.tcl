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


catch {rename NrrdTextureBuilder ""}

itcl_class SCIRun_Visualization_NrrdTextureBuilder {
    inherit Module
    constructor {config} {
	set name NrrdTextureBuilder
	set_defaults
    }
    method set_defaults {} {
	global $this-card_mem
	global $this-card_mem_auto
	set $this-card_mem 16
	set $this-card_mem_auto 1
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

	frame $w.f.memframe -relief groove -border 2
	label $w.f.memframe.l -text "Graphics Card Memory (MB)"
	pack $w.f.memframe -side top -padx 2 -pady 2 -fill both
	pack $w.f.memframe.l -side top -fill x
	checkbutton $w.f.memframe.auto -text "Autodetect" -relief flat \
            -variable $this-card_mem_auto -onvalue 1 -offvalue 0 \
            -anchor w -command "$s; $n"
	pack $w.f.memframe.auto -side top -fill x

	frame $w.f.memframe.bf -relief flat -border 2
        set bf $w.f.memframe.bf
	pack $bf -side top -fill x
	radiobutton $bf.b0 -text 4 -variable $this-card_mem -value 4 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b1 -text 8 -variable $this-card_mem -value 8 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b2 -text 16 -variable $this-card_mem -value 16 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b3 -text 32 -variable $this-card_mem -value 32 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b4 -text 64 -variable $this-card_mem -value 64 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b5 -text 128 -variable $this-card_mem -value 128 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b6 -text 256 -variable $this-card_mem -value 256 \
	    -command $n -state disabled -foreground darkgrey
	radiobutton $bf.b7 -text 512 -variable $this-card_mem -value 512 \
	    -command $n -state disabled -foreground darkgrey
	pack $bf.b0 $bf.b1 $bf.b2 $bf.b3 $bf.b4 $bf.b5 $bf.b6 $bf.b7 -side left -expand yes\
                -fill x
	
	$this state

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }
    method set_card_mem {mem} {
	set $this-card_mem $mem
    }
    method set_card_mem_auto {auto} {
	set $this-card_mem_auto $auto
    }
    method state {} {
	set w .ui[modname]
	if {[winfo exists $w] == 0} {
	    return
	}
	if {[set $this-card_mem_auto] == 1} {
            $this deactivate $w.f.memframe.bf.b0
            $this deactivate $w.f.memframe.bf.b1
            $this deactivate $w.f.memframe.bf.b2
            $this deactivate $w.f.memframe.bf.b3
            $this deactivate $w.f.memframe.bf.b4
            $this deactivate $w.f.memframe.bf.b5
            $this deactivate $w.f.memframe.bf.b6
            $this deactivate $w.f.memframe.bf.b7
	} else {
            $this activate $w.f.memframe.bf.b0
            $this activate $w.f.memframe.bf.b1
            $this activate $w.f.memframe.bf.b2
            $this activate $w.f.memframe.bf.b3
            $this activate $w.f.memframe.bf.b4
            $this activate $w.f.memframe.bf.b5
            $this activate $w.f.memframe.bf.b6
            $this activate $w.f.memframe.bf.b7
        }
    }
    method activate { w } {
	$w configure -state normal -foreground black
    }
    method deactivate { w } {
	$w configure -state disabled -foreground darkgrey
    }
}
