#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
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
	set $this-firstwidth 12

	# for entire nrrd
	global $this-type
	global $this-dimension

        # for each axis
	global $this-size1	
	global $this-size2	
	global $this-size3	

	global $this-center1	
	global $this-center2	
	global $this-center3	

	global $this-label0
	global $this-label1	
	global $this-label2	
	global $this-label3	

	global $this-spacing1	
	global $this-spacing2	
	global $this-spacing3

	global $this-min1	
	global $this-min2	
	global $this-min3

	global $this-max1	
	global $this-max2	
	global $this-max3	

	global $this-active_tab

	set $this-active_tab "Tuple Axis"
	# these won't be saved 
	set $this-type "---"
	set $this-dimension "---"
	set $this-size1	 "---"
	set $this-size2	 "---"
	set $this-size3	 "---"
	set $this-center1 "---"	
	set $this-center2 "---"	
	set $this-center3 "---"	
	set $this-label0 "---"
	set $this-label1 "---"	
	set $this-label2 "---"	
	set $this-label3 "---"	
	set $this-spacing1 "---"	
	set $this-spacing2 "---"	
	set $this-spacing3 "---"
	set $this-min1 "---"	
	set $this-min2 "---"	
	set $this-min3 "---"
	set $this-max1 "---"	
	set $this-max2 "---"	
	set $this-max3 "---"

    }

    method set_active_tab {act} {
	global $this-active_tab
	#puts stdout $act
	set $this-active_tab $act
    }

    method switch_to_active_tab {name1 name2 op} {
	#puts stdout "switching"
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set axis_frame [$w.att.axis_info childsite]
	    $axis_frame.tabs view [set $this-active_tab]
	}
    }

    method fill_tuple_tab {} {
        set w .ui[modname]
	set att [$w.att childsite]
	# tuple tab is always first
	set tuple [$att.tabs childsite 0]
	#parse the label axis string to gather all the info out of it.
	set l [regexp -inline -all -- {(\w+):(\w+),?} [set $this-label0]]

	set last [$tuple.listbox index end]
	$tuple.listbox delete 0 $last
	$tuple.listbox insert 0 "Name (Type)"
	
	if {[llength $l]} {
	    set ind 1
	    foreach {match name type} $l {
		$tuple.listbox insert $ind "$name ($type)"
		incr ind
	    }
	} else {
	    $tuple.listbox insert 1 "unknown tuple axis label format"
	    $tuple.listbox insert 2 [set $this-label0]
	}
    }

    method add_tuple_tab {af} {

	set tuple [$af.tabs add -label "Tuple Axis" \
		       -command "$this set_active_tab \"Tuple Axis\""]

	iwidgets::scrolledlistbox $tuple.listbox -vscrollmode static \
	    -hscrollmode dynamic -scrollmargin 3 -height 60

	fill_tuple_tab
	pack $tuple.listbox -side top -fill both -expand 1
    }

    method add_axis1_tab {af} {

	set a1 [$af.tabs add -label "Axis 1" \
		       -command "$this set_active_tab \"Axis 1\""]

	labelpair $a1.l "Label" $this-label1
	labelpair $a1.c "Center" $this-center1
	labelpair $a1.sz "Size" $this-size1
	labelpair $a1.sp "Spacing" $this-spacing1
	labelpair $a1.mn "Min" $this-min1
	labelpair $a1.mx "Max" $this-max1

	pack $a1.l $a1.c $a1.sz  $a1.sp $a1.mn $a1.mx  -side top
	pack $a1 -side top -fill both -expand 1
    }

    method add_axis2_tab {af} {

	set a2 [$af.tabs add -label "Axis 2" \
		       -command "$this set_active_tab \"Axis 2\""]

	labelpair $a2.l "Label" $this-label2
	labelpair $a2.c "Center" $this-center2
	labelpair $a2.sz "Size" $this-size2
	labelpair $a2.sp "Spacing" $this-spacing2
	labelpair $a2.mn "Min" $this-min2
	labelpair $a2.mx "Max" $this-max2

	pack $a2.l $a2.c $a2.sz  $a2.sp $a2.mn $a2.mx  -side top
	pack $a2 -side top -fill both -expand 1
    }

    method add_axis3_tab {af} {

	set a3 [$af.tabs add -label "Axis 3" \
		       -command "$this set_active_tab \"Axis 3\""]

	labelpair $a3.l "Label" $this-label3
	labelpair $a3.c "Center" $this-center3
	labelpair $a3.sz "Size" $this-size3
	labelpair $a3.sp "Spacing" $this-spacing3
	labelpair $a3.mn "Min" $this-min3
	labelpair $a3.mx "Max" $this-max3

	pack $a3.l $a3.c $a3.sz  $a3.sp $a3.mn $a3.mx  -side top
	pack $a3 -side top -fill both -expand 1
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	iwidgets::labeledframe $w.att -labelpos nw \
		               -labeltext "Nrrd Attributes" 
			       
	pack $w.att 
	set att [$w.att childsite]
	
	labelpair $att.type "C Type" $this-type
	labelpair $att.dim "Dimension" $this-dimension

	# notebook for the axis specific info.
	iwidgets::tabnotebook  $att.tabs -height 350 -width 325 \
	    -raiseselect true 

	add_tuple_tab $att
	add_axis1_tab $att
	add_axis2_tab $att
	add_axis3_tab $att

	# view the active tab
	$att.tabs view [set $this-active_tab]	
	$att.tabs configure -tabpos "n"

	pack $att.type $att.dim -side top 
	pack $att.tabs -side top -fill x -expand yes

	frame $w.exec
	pack $w.exec -side bottom -padx 5 -pady 5
	button $w.exec.execute -text "Execute" -command "$this-c needexecute"
	pack $w.exec.execute -side top -e n
    }


    method labelpair { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar $text2 -width 40 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon $win.l2 -side left
    } 

}




