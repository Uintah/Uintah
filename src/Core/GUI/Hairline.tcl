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


#  Hairline.tcl
#  Written by:
#   Yarden Livnat
#   Department of Computer Science
#   University of Utah
#   July 2001
#  Copyright (C) 2001 SCI Group

package require Iwidgets 3.0

itcl::class Hairline {

    variable opt
    variable menu
    variable gl
    variable parent
    variable n
    variable initialized
    variable current
    variable allocated

    constructor { args } {
	set initialized 0
	set val(0) 1
	set allocated 0
	set current 0
    }

    destructor {
	delete object  $w.hairline
    }

    method ui { window name } {
	set opt $window.$name
	set n $name

    }

    method init {} {
	iwidgets::labeledframe $opt -labeltext $n -labelpos nw 
	pack $opt
	set initialized 1
    }
	
    method add { n color value} {

	if { $initialized == 0 } {
	    $this init
	}

	set w [$opt childsite].$n
	    
 	frame $w-line -background $color -width 10 -height 2
 	label $w-val -text $value
#	grid $w-line $w-val -padx 3 -sticky w
    }


    method widget { name } {
	$this-c widget $name
    }

    method values { args } {
	if { $initialized == 0 } {
	    $this init
	}
  	set w [$opt childsite]

	set l [llength $args]
	if { $allocated < $l } {
	    for {set i $allocated} {$i < $l} {incr i} {
		frame $w.$i-line -width 10 -height 2
		label $w.$i-val 
	    }
	    set allocated $l
	}
	
	if { $l < $current } {
	    for {set i $l} {$i < $current} {incr i} {
		grid forget $w.$i-line $w.$i-val
	    }
	} 

	if { $current < $l } {
	    for {set i $current} {$i < $l} {incr i} {
		grid $w.$i-line $w.$i-val -padx 3 -sticky w 
	    }
	} 
	    

  	for {set i 0} {$i < $l} {incr i} {
	    set item [lindex $args $i]
	    set c [lindex $item 0]
	    set v [lindex $item 1]
  	    $w.$i-line config -bg $c
  	    $w.$i-val config -text $v
  	}

	set current $l
    }
}
