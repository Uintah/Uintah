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
	global $this-type
	global $this-dimension
	global $this-active_tab
	global $this-label0

	# these won't be saved 
	set $this-firstwidth 12
	set $this-type "---"
	set $this-dimension 0
	set $this-active_tab "Tuple Axis"
	set $this-label0 "---"
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

	if {![winfo exists $w]} {
            return
        }
	set att [$w.att childsite]
	# tuple tab is always first
	set tuple [$att.tabs childsite 0]
	#parse the label axis string to gather all the info out of it.
	set l [regexp -inline -all -- {([\w-]+):(\w+),?} [set $this-label0]]

	set last [$tuple.listbox index end]
	$tuple.listbox delete 0 $last
	$tuple.listbox insert 0 "Name (Type)"
	
	if {[llength $l]} {
	    set ind 1
	    foreach {match axname type} $l {
		$tuple.listbox insert $ind "$axname ($type)"
		incr ind
	    }
	} else {
	    $tuple.listbox insert 1 "unknown tuple axis label format"
	    $tuple.listbox insert 2 [set $this-label0]
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
	    set_active_tab "Tuple Axis"
	    $af.tabs view [set $this-active_tab]
	}
    }

    method add_tabs {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    set af [$w.att childsite]

	    for {set i 1} {$i < [set $this-dimension]} {incr i} {
		set t [$af.tabs add -label "Axis $i" \
			     -command "$this set_active_tab \"Axis $i\""]
		
		labelpair $t.l "Label" $this-label$i
		labelpair $t.c "Center" $this-center$i
		labelpair $t.sz "Size" $this-size$i
		labelpair $t.sp "Spacing" $this-spacing$i
		labelpair $t.mn "Min" $this-min$i
		labelpair $t.mx "Max" $this-max$i
		
		pack $t.l $t.c $t.sz  $t.sp $t.mn $t.mx  -side top
		pack $t -side top -fill both -expand 1
	    }
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

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	# notebook for the axis specific info.
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
	add_tabs

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




