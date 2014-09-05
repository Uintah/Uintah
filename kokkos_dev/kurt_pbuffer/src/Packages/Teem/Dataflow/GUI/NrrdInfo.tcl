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

#    File   : NrrdInfo.tcl
#    Author : Martin Cole
#    Date   : Wed Feb  5 11:45:16 2003

itcl_class Teem_NrrdData_NrrdInfo {
    inherit Module
    constructor {config} {
        set name NrrdInfo
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	global $this-name
	global $this-type
	global $this-dimension
	global $this-active_tab
	global $this-label0
	global $this-kind0
	global $this-center0
	global $this-size0
	global $this-spacing0
	global $this-min0
	global $this-max0
	global $this-initialized
		    
	# these won't be saved 
	set $this-firstwidth 12
	set $this-name "---"
	set $this-type "---"
	set $this-dimension 0
	set $this-active_tab "Axis 0"
	set $this-label0 "---"
	set $this-kind0 "---"
	set $this-center0 "---"
	set $this-size0 "---"
	set $this-spacing0 "---"
	set $this-min0 "---"
	set $this-max0 "---"
	set $this-initialized 0
    }

    method set_active_tab {act} {
	global $this-active_tab
	# puts stdout $act
	set $this-active_tab $act
    }

    method switch_to_active_tab {name1 name2 op} {
	# puts stdout "switching"
	set window .ui[modname]
	if {[winfo exists $window.att.axis_info]} {
	    set axis_frame [$w.att.axis_info childsite]
	    $axis_frame.tabs view [set $this-active_tab]
	}
    }

    method delete_tabs {} {
	set w .ui[modname]
	
	if {[winfo exists $w.att]} {
	    set af [$w.att childsite]
	    set l [$af.tabs childsite]
	    if { [llength $l] > 1 } { 
		$af.tabs delete 1 end
	    }
	    set_active_tab "Axis 0"
	    $af.tabs view [set $this-active_tab]

	    # Clear Axis 0 and dimension
	    set $this-dimension 0
	    set $this-label0 "---"
	    set $this-kind0 "---"
	    set $this-center0 "---"
	    set $this-size0 "---"
	    set $this-spacing0 "---"
	    set $this-min0 "---"
	    set $this-max0 "---"
	}
    }
    
    method add_tabs {} {
	set w .ui[modname]
        if {[winfo exists $w]} {

	    set af [$w.att childsite]
	    
	    if {[set $this-dimension] != 0} {
		# CHANGED FROM {set i 1}
		if {[set $this-initialized] == 0} {
		    set t [$af.tabs add -label "Axis 0" \
			       -command "$this set_active_tab \"Axis 0\""]
		    
		    labelpair $t.l "Label" $this-label0
		    labelpair $t.k "Kind" $this-kind0
		    labelpair $t.c "Center" $this-center0
		    labelpair $t.sz "Size" $this-size0
		    labelpair $t.sp "Spacing" $this-spacing0
		    labelpair $t.mn "Min" $this-min0
		    labelpair $t.mx "Max" $this-max0
		    
		    pack $t.l $t.k $t.c $t.sz  $t.sp $t.mn $t.mx  -side top \
			-fill x 
		    pack $t -side top -fill both -expand 1	
		    set $this-initialized 1
		}

		for {set i 1} {$i < [set $this-dimension]} {incr i} {
		    set t [$af.tabs add -label "Axis $i" \
			       -command "$this set_active_tab \"Axis $i\""]
		    
		    labelpair $t.l "Label" $this-label$i
		    labelpair $t.k "Kind" $this-kind$i
		    labelpair $t.c "Center" $this-center$i
		    labelpair $t.sz "Size" $this-size$i
		    labelpair $t.sp "Spacing" $this-spacing$i
		    labelpair $t.mn "Min" $this-min$i
		    labelpair $t.mx "Max" $this-max$i
		    
		    pack $t.l $t.k $t.c $t.sz  $t.sp $t.mn $t.mx  -side top \
			-fill x
		    pack $t -side top -fill both -expand 1
		} 
	    } else {
		# Build Axis 0 tab as empty
		set t [$af.tabs add -label "Axis 0" \
			   -command "$this set_active_tab \"Axis 0\""]
		    
		labelpair $t.l "Label" $this-label0
		labelpair $t.k "Kind" $this-kind0
		labelpair $t.c "Center" $this-center0
		labelpair $t.sz "Size" $this-size0
		labelpair $t.sp "Spacing" $this-spacing0
		labelpair $t.mn "Min" $this-min0
		labelpair $t.mx "Max" $this-max0
		
		pack $t.l $t.k $t.c $t.sz  $t.sp $t.mn $t.mx  -side top \
		    -fill x
		pack $t -side top -fill both -expand 1	
		set $this-initialized 1
	    }
	}
    }
    
    #     method add_tuple_tab {af} {
    
    # 	set tuple [$af.tabs add -label "Tuple Axis" \
    # 		       -command "$this set_active_tab \"Tuple Axis\""]
    
    # 	iwidgets::scrolledlistbox $tuple.listbox -vscrollmode static \
    # 	    -hscrollmode dynamic -scrollmargin 3 -height 60
    
    # 	fill_tuple_tab
    # 	pack $tuple.listbox -side top -fill both -expand 1
    #     }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	
	# notebook for the axis specific info.
	iwidgets::labeledframe $w.att -labelpos nw \
	    -labeltext "Nrrd Attributes" 
	
	pack $w.att -expand y -fill both
	set att [$w.att childsite]
	
	labelpair $att.name "Name" $this-name
	labelpair $att.type "C Type" $this-type
	labelpair $att.dim "Dimension" $this-dimension
	
	# notebook for the axis specific info.
	iwidgets::tabnotebook  $att.tabs -height 350 -width 325 \
	    -raiseselect true 
	
	# add_tuple_tab $att
	add_tabs
	
	# view the active tab
	$att.tabs view [set $this-active_tab]	
	$att.tabs configure -tabpos "n"
	
	pack $att.name $att.type $att.dim -side top -anchor nw \
	    -expand yes -fill x
	pack $att.tabs -side top -fill x -expand yes
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
    }


    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 40 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon -side left
	pack $win.l2 -side left -fill x -expand yes
    } 

}




