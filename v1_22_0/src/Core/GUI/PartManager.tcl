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


#
#  PartManager.tcl
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   Oct 2001
#
#  Copyright (C) 2001 SCI Group
#

package require Iwidgets 3.0

itcl::class PartManager {

    variable w
    variable n

    constructor {} {
	set n 0
    }

    method ui { window } {
	set w $window

	iwidgets::tabnotebook  $w.tabs -raiseselect true -auto true -width 400
	$w.tabs configure -tabpos "n" 
	pack $w.tabs -side top -anchor w -expand true -fill both
    }

    method lock {} {
	set current [$w.tabs view]
	puts "lock $current"
	for {set i 0} {$i < $n} {incr i} {
	    if { $i != $current } {
		$w.tabs pageconfigure $i -state disable
	    }
	}
    }

    method unlock {} {
	for {set i 0} {$i < $n} {incr i} {
	    $w.tabs pageconfigure $i -state normal
	}
    }

    method new-child-window { name } {
	set child [$w.tabs add -label "$name" -command "$this-c select $n"]
	incr n
	return $child
    }
}
	
