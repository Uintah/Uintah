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
#  OpenGLWindow
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   July 2001
#
#  Copyright (C) 2001 SCI Group
#

itcl::class OpenGLWindow {

    variable w
    variable prev

    constructor {} {
    }

    method ui { window parent } {
	set w $window

	opengl $w -geometry 500x300 -doublebuffer true \
	    -direct true -rgba true 
	bind $w <Map> "$this-c map $w"
	bind $w <Expose> "$parent-c redraw $w"
	bind $w <Enter> { set prev [focus]; focus %W }
	bind $w <Leave> { focus $prev }
	pack $w
    }

    method change_params { n } {
    }
	
    method setobj { obj } {
	bindtags $w [list [$obj getbinds] $w all]
    }

    method resize { width height } {
	$w configure -geometry ${width}x$height
    }

    method set-cursor { c } {
	set current [$w cget -cursor]
	$w configure -cursor $c
	return $current
    }


    method add-bind { b } {
	bindtags $w [concat $b [bindtags $w] ]
    }

    method rem-bind { b } {
	bindtags $w [ldelete $b [bindtags $w] ]
    }

    method ldelete { item list } {
	set i [lsearch -exact $list $item]
	if { $i > -1 } {
	    return [lreplace $list $i $i]
	} else {
	    return $list
	}
    }
}
	
	
