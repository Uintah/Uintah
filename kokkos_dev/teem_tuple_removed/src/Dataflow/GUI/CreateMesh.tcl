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

itcl_class SCIRun_FieldsCreate_CreateMesh {
    inherit Module
    constructor {config} {
        set name CreateMesh
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	global $this-fieldname
	global $this-meshname
	global $this-fieldbasetype
	global $this-datatype
	set $this-meshname "Created Mesh"
	set $this-fieldname "Created Field"
	set $this-fieldbasetype "TetVolField"
	set $this-datatype "double"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w
	wm maxsize $w 397 187 

	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Field Type" 
			       
	pack $w.att 
	set att [$w.att childsite]
	
	labelpair $att.l1 "Name" $this-fldname
	labelpair $att.l2 "Typename" $this-inputdatatype
	pack $att.l1 $att.l2 -side top 

	iwidgets::Labeledframe $w.fbt -labelpos nw \
		               -labeltext "Output Mesh Type"
	pack $w.fbt
	set fbt [$w.fbt childsite]
	labelcombo $fbt.l1 "Mesh Type" \
	    { \
		  CurveField \
		  HexVolField \
		  PointCloudField \
		  PrismVolField \
		  QuadSurfField \
		  TetVolField \
		  TriSurfField \
	      } \
	    $this-fieldbasetype

	pack $fbt.l1 -side top 

	iwidgets::Labeledframe $w.edit -labelpos nw \
		               -labeltext "Output Field Type" 
	pack $w.edit 
	set edit [$w.edit childsite]
	labelcombo $edit.l1 "Data Type" \
		{"unsigned char" "unsigned short" "unsigned int" \
		char short int float double Vector Tensor} \
		   $this-datatype
	pack $edit.l1 -side top 

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


    method labelcombo { win text1 arglist var} {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
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

	label $win.l2 -text "" -width 40 -anchor w -just left

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

    method config_labelcombo { win arglist sel} {
	if {![winfo exists $win]} {
	    return
	}
	$win.c delete 0 end
	if {[llength $arglist]==0} {
	    $win.c insert end ""
	}
	set i 0
	set length [llength $arglist]
	for {set elem [lindex $arglist $i]} {$i<$length} \
	    {incr i 1; set elem [lindex $arglist $i]} {
	    $win.c insert end $elem
	}
	
	if {"$sel"!="---"} {
	    $win.c select $sel
	}
    }
}




