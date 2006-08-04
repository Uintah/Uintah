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
#  Graph.tcl
#
#  Written by:
#   Yarden Livant
#   Department of Computer Science
#   University of Utah
#   July 2001
#
#  Copyright (C) 2001 SCI Group
#

package require Iwidgets 3.0

itcl::class Graph {

    variable graph
    variable parent-window
    variable status

    constructor {} {
    }

    method ui { w } {
	set parent-window $w

	# menubar
	iwidgets::menubar $w.menu 
	$w.menu add menubutton .opt -text "Opt"


	# UIs
	frame $w.ui

	# status
	label $w.status -textvariable status 

	frame $w.f
	iwidgets::toolbar $w.f.tb -helpvariable status -orient vertical

	# graph area
	
	frame $w.f.graph

	pack $w.f.tb  -fill y -side left
	pack $w.f.graph -expand yes  -fill both

	# Info winfow
	#iwidgets::Labeledframe $w.info -labelpos nw -labeltext "Info"
	
	# pack $w.info 

	pack $w.menu -fill x 
	pack $w.ui -anchor w 
	pack $w.f -expand yes -fill both
	pack $w.status  -fill x
	
    }

}
	

	#
	# Graph
	#
#	iwidgets::scrolledframe $w.f.graph -width 450 -height 350 \
\#	    -hscrollmode dynamic -vscrollmode dynamic 

	#
	#
	# OpenGL window of Graph
	#  
#	set $this-gl-window [$w.f.graph childsite].gl

	
