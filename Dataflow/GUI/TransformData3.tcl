#
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

itcl_class SCIRun_Fields_TransformData3 {
    inherit Module
    constructor {config} {
        set name TransformData3
	
	global $this-function
	global $this-outputdatatype

        set_defaults
    }

    method set_defaults {} {
	set $this-function "result = v * 10;"
	set $this-outputdatatype "input 0"
    }

    method update_text {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    set $this-function [$w.row1 get 1.0 end]
        }
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	labelcombo $w.otype "Output Data Type" \
	    {"input 0" "input 1" "input 2" "unsigned char" "unsigned short" \
		 "unsigned int" \
		 char short int float double Vector Tensor} \
	    $this-outputdatatype

	option add *textBackground white	
	iwidgets::scrolledtext $w.row1 -height 60 -hscrollmode dynamic

	bind $w.row1 <Leave> "$this update_text"
	$w.row1 insert end [set $this-function]

	frame $w.row2
	button $w.row2.execute -text "Execute" -command "$this-c needexecute"
	pack $w.row2.execute -side left -e y -f both -padx 5 -pady 5
	pack $w.row1 $w.row2 -side top -e y -f both -padx 5 -pady 5
    }


    method labelcombo { win text1 arglist var} {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left
	iwidgets::optionmenu $win.c -foreground darkred \
		-command " $this comboget $win.c $var "

	set i 0
	set found 0
	set length [llength $arglist]
	for {set elem [lindex $arglist $i]} {$i<$length} \
	    {incr i 1; set elem [lindex $arglist $i]} {
	    if {"$elem"=="[set $var]"} {
		set found 1
	    }
	    $win.c insert end $elem
	}

	if {!$found} {
	    $win.c insert end [set $var]
	}

	label $win.l2 -text "" -width 20 -anchor w -just left

	# hack to associate optionmenus with a textvariable
	bind $win.c <Map> "$win.c select {[set $var]}"

	pack $win.l1 $win.colon -side left
	pack $win.c $win.l2 -side left	
    }

    method comboget { win var } {
	if {![winfo exists $win]} {
	    return
	}
	if { "$var"!="[$win get]" } {
	    set $var [$win get]
	}
    }
}


