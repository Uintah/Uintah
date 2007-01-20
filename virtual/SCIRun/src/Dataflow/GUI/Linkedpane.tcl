#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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
#  LinkedPane.tcl
#
#  Written by:
#   McKay Davis
#   Scientific Computing and Imaging Institute
#   University of Utah
#   December 2005
#
#  Copyright (C) 2005 SCI Group
#


itk::usual LinkedPane {
    keep -background -cursor -sashcursor
}

# Linked Pane is just a iwidgets::Panedwindow that is linked to another one while moving
class LinkedPane {
    inherit ::iwidgets::Panedwindow
    private variable other ""
    constructor { args } { 
        eval ::iwidgets::Panedwindow::constructor $args
    }
    public method set_link { w } { set other $w }

    protected method _startGrip {where num { from_link 0 } } {
	if { !$from_link && [winfo exists $other] } { 
	    $other _startGrip $where $num 1
	}
        ::iwidgets::Panedwindow::_startGrip $where $num
    }

    protected method _endGrip {where num {from_link 0}} {
	if { !$from_link && [winfo exists $other] } { 
	    $other _endGrip $where $num 1
	}
	::iwidgets::Panedwindow::_endGrip $where $num
    }

    protected method _moveSash {where num {from_link 0}} {
	if { !$from_link && [winfo exists $other] } { 
	    $other _moveSash $where $num 1
	}
	::iwidgets::Panedwindow::_moveSash $where $num
    }
}

# This proc is just to make sure this file shows up in the tclIndex
proc Linkedpane {args} {
    return [uplevel 1 LinkedPane $args]
}
