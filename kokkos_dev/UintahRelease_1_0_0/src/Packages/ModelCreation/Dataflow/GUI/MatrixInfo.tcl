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


itcl_class ModelCreation_DataInfo_MatrixInfo {
    inherit Module
    constructor {config} {
        set name MatrixInfo
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	# these won't be saved 
	global $this-matrixname
	global $this-generation
	global $this-typename
	global $this-rows
	global $this-cols
	global $this-elements
	set $this-matrixname "---"
	set $this-generation "---"
	set $this-typename "---"
	set $this-rows "---"
	set $this-cols "---"
	set $this-elements "---"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Matrix Attributes" 
			       
	pack $w.att -fill x
	set att [$w.att childsite]
	
	entrypair $att.l1 "Name" $this-matrixname
	entrypair $att.l2 "Generation" $this-generation
	labelpair $att.l3 "Typename" $this-typename
	labelpair $att.l7 "# Rows" $this-rows
	labelpair $att.l8 "# Columns" $this-cols
	labelpair $att.l9 "# Elements" $this-elements
	pack $att.l1 $att.l2 $att.l3 \
	     $att.l7 $att.l8 $att.l9 -side top -expand y -fill x

	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }

    method entrypair { win text1 text2 } {

	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon -text ":" -width 2 -anchor w -just left 

	entry $win.l2 -textvar $text2 \
	    -just left -width 40 \
	    -relief flat -state disabled \
	    -fore darkred -borderwidth 0 \
	    -xscrollcommand [list $win.xscroll set]

	scrollbar $win.xscroll -orient horizontal \
	    -command [list $win.l2 xview]

	pack $win.l1 $win.colon $win.l2 -side left
	pack $win.xscroll -side left -fill x
    } 

    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -anchor w -just left \
	    -fore darkred -borderwidth 0
	pack $win.l1 $win.colon $win.l2 -side left
    } 

    method labelpair2 { win text1 text2x text2y } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ": " -width 2 -anchor w -just left
	label $win.l2x -textvar $text2x -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma1  -text ", " -anchor w -just left  \
	    -fore darkred -borderwidth 0
	label $win.l2y -textvar $text2y -anchor w -just left \
	    -fore darkred -borderwidth 0
	pack $win.l1 $win.colon \
	    $win.l2x $win.comma1 $win.l2y -side left -padx 0
    } 

    method labelpair3 { win text1 text2x text2y text2z } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ": " -width 2 -anchor w -just left
	label $win.l2x -textvar $text2x -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma1  -text ", " -anchor w -just left  \
	    -fore darkred -borderwidth 0
	label $win.l2y -textvar $text2y -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.comma2  -text ", " -anchor w -just left \
	    -fore darkred -borderwidth 0
	label $win.l2z -textvar $text2z -anchor w -just left \
	    -fore darkred -borderwidth 0
	pack $win.l1 $win.colon \
	    $win.l2x $win.comma1 $win.l2y $win.comma2 $win.l2z \
	    -side left -padx 0
    } 
}




