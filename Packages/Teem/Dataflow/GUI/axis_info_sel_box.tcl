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

#    File   : axis_info_sel_box.tcl
#    Author : Martin Cole
#    Date   : Fri Feb 21 10:16:34 2003

global teem_num_axes
set teem_num_axes 0


proc get_selection {w} {
    set f [$w childsite]
    $f.rb get
}

proc make_axis_info_sel_box {w command} {
    
    iwidgets::scrolledframe $w -relief groove -width 250 -height 200 \
		-labelpos nw -labeltext "Axis Info and Selection"

    pack $w -expand yes -fill both -side top -padx 8
    set f [$w childsite]

    iwidgets::radiobox $f.rb -relief flat -command $command
    pack $f.rb -side top -fill x -fill y -expand yes -padx 4 -pady 4
}

proc add_axis {w tag info} {
    global teem_num_axes
    set f [$w childsite]
    $f.rb add $tag -text $info
    incr teem_num_axes
}

proc select_axis {w tag} {
    global teem_num_axes
    set f [$w childsite]
    $f.rb select $tag
}

proc delete_all_axes {w} {
    global teem_num_axes
    set f [$w childsite]
    # if catch catches an error its already empty
    if { [catch { set last [$f.rb index end] } ] } {
	set last 0
    } else {
	set i 0
	while {$i <= $last} {
	    $f.rb delete 0
	    incr i
	} 
    }
}