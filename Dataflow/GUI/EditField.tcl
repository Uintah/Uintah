itcl_class SCIRun_Fields_EditField {
    inherit Module
    constructor {config} {
        set name EditField
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	# these won't be saved 
	global $this-fldname
	global $this-typename
	global $this-datamin
	global $this-datamax
	global $this-numnodes
	global $this-numelems
	global $this-dataat
	global $this-minx
	global $this-miny
	global $this-minz
	global $this-maxx
	global $this-maxy
	global $this-maxz
	set $this-fldname "---"
	set $this-typename "---"
	set $this-datamin "---"
	set $this-datamax "---"
	set $this-numnodes "---"
	set $this-numelems "---"
	set $this-dataat "---"
	set $this-minx "---"
	set $this-miny "---"
	set $this-minz "---"
	set $this-maxx "---"
	set $this-maxy "---"
	set $this-maxz "---"

	# these will be saved
	global $this-fldname2
	global $this-typename2
	global $this-datamin2
	global $this-datamax2
	global $this-dataat2
	global $this-cfldname
	global $this-ctypename
	global $this-cdataminmax
	global $this-cbbox
	global $this-cnumelems
	global $this-cdataat
	global $this-minx2
	global $this-miny2
	global $this-minz2
	global $this-maxx2
	global $this-maxy2
	global $this-maxz2
	set $this-fldname2 ""
	set $this-typename2 "--- No typename ---"
	set $this-datamin2 0
	set $this-datamax2 0
	set $this-dataat2 "Nodes"
	set $this-cfldname 0
	set $this-ctypename 0
	set $this-cdataminmax 0
	set $this-cbbox 0
	set $this-cnumelems 0
	set $this-cdataat 0
	set $this-minx2 1
	set $this-miny2 1
	set $this-minz2 1
	set $this-maxx2 0
	set $this-maxy2 0
	set $this-maxz2 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	wm minsize $w 416 408
	wm maxsize $w 416 1000

	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Field Attributes" 
			       
	pack $w.att 
	set att [$w.att childsite]
	
	labelpair $att.l1 "Name" $this-fldname
	labelpair $att.l2 "Typename" $this-typename
	labelpairmulti $att.l3 "BBox min" "[set $this-minx], \
		               [set $this-miny], [set $this-minz]"
	labelpairmulti $att.l4 "BBox max" "[set $this-maxx], \
                               [set $this-maxy], [set $this-maxz]"
	labelpairmulti $att.l5 "Data min,max" "[set $this-datamin], \
		                          [set $this-datamax]"
	labelpair $att.l7 "# Nodes" $this-numnodes
	labelpair $att.l8 "# Elements" $this-numelems
	labelpair $att.l9 "Data at" $this-dataat
	pack $att.l1 $att.l2 $att.l3 $att.l4 $att.l5 \
	     $att.l7 $att.l8 $att.l9 -side top 

	button $w.copy -text "copy input to output" \
                       -command "$this copy_attributes"
	pack $w.copy -side top -pady 5

	iwidgets::Labeledframe $w.edit -labelpos nw \
		               -labeltext "Output Field Attributes" 
			       
	pack $w.edit 
	set edit [$w.edit childsite]
	
	labelentry $edit.l1 "Name" $this-fldname2 $this-cfldname
	labelcombo $edit.l2 "Typename" \
		   "[possible_typenames [set $this-typename2]]" \
                   $this-typename2 $this-ctypename
	labelentry3 $edit.l3 "BBox min" $this-minx2 $this-miny2 \
		    $this-minz2 $this-cbbox "$this-c update_widget"
	labelentry3 $edit.l4 "BBox max" $this-maxx2 $this-maxy2 $this-maxz2 \
		    $this-cbbox "$this-c update_widget"
	labelentry2 $edit.l5 "Data min,max" $this-datamin2 $this-datamax2 \
		    $this-cdataminmax
	labelcombo $edit.l9 "Data at" {Nodes Edges Faces Cells} \
		   $this-dataat2 $this-cdataat
	pack $edit.l1 $edit.l2 $edit.l5 \
	     $edit.l9 -side top 

    }

    method update_multifields {} {
        set w .ui[modname]
	if {![winfo exists $w]} {
	    return
	}
	set att [$w.att childsite]
	$att.l3.l2 configure -text "[set $this-minx], [set $this-miny], \
		                  [set $this-minz]"
	$att.l4.l2 configure -text "[set $this-maxx], [set $this-maxy], \
		                  [set $this-maxz]"
	$att.l5.l2 configure -text "[set $this-datamin], [set $this-datamax]"
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

    method labelpairmulti { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -text $text2 -width 40 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon $win.l2 -side left
    } 

    method labelentry { win text1 text2 var } {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var $var
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	entry $win.l2 -width 40 -just left \
		-fore darkred -text $text2
	pack $win.b $win.l1 $win.colon -side left
	pack $win.l2 -padx 5 -side left
    }

    method labelentry2 { win text1 text2 text3 var } {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var $var
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	entry $win.l2 -width 10 -just left \
		-fore darkred -text $text2
	entry $win.l3 -width 10 -just left \
		-fore darkred -text $text3
	label $win.l4 -width 40
	pack $win.b $win.l1 $win.colon -side left
	pack $win.l2 $win.l3 $win.l4 -padx 5 -side left
    }

    method labelentry3 { win text1 text2 text3 text4 var func} {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var $var
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	entry $win.l2 -width 10 -just left \
		-fore darkred -text $text2
	entry $win.l3 -width 10 -just left \
		-fore darkred -text $text3
	entry $win.l4 -width 10 -just left \
		-fore darkred -text $text4
	label $win.l5 -width 40
	pack $win.b $win.l1 $win.colon -side left
	pack $win.l2 $win.l3 $win.l4 $win.l5 -padx 5 -side left
	
	bind $win.l2 <Return> $func
	bind $win.l3 <Return> $func
	bind $win.l4 <Return> $func
    }

    method labelcombo { win text1 arglist var var2} {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var $var2
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

	pack $win.b $win.l1 $win.colon -side left
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

    method labeloption { win text1 text2 text3 var var2} {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var $var2
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
                      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	radiobutton $win.l2 -text $text2 -var $var -val $text2 -fore darkred
	radiobutton $win.l3 -text $text3 -var $var -val $text3 -fore darkred
	label $win.l4 -width 40
	pack $win.b $win.l1 $win.colon -side left
        pack $win.l2 $win.l3 $win.l4 -side left
    }

    method copy_attributes {} {
	set w .ui[modname]
	if {![winfo exists $w]} {
	    return
	}
	set att [$w.att childsite]
	set edit [$w.edit childsite]
	if {"[set $this-fldname]"!="--- Name Not Assigned ---"} {
	    set $this-fldname2 [set $this-fldname]
	} else {
	    set $this-fldname2 ""
	}
	config_labelcombo $edit.l9 {Nodes Edges Faces Cells} [set $this-dataat]
	set $this-datamin2 [set $this-datamin]
	set $this-datamax2 [set $this-datamax]
	set $this-minx2 [set $this-minx]
	set $this-miny2 [set $this-miny]
	set $this-minz2 [set $this-minz]
	set $this-maxx2 [set $this-maxx]
	set $this-maxy2 [set $this-maxy]
	set $this-maxz2 [set $this-maxz]
	config_labelcombo $edit.l2 [possible_typenames [set $this-typename]] \
                          [set $this-typename]
    }

    method possible_typenames { type } {
	set index1 [string first "<" $type]
	set index2 $index1
	set name1 ""
	set name2 ""

	if {$index1<0} {
	    return ""
	} else {
	    set name1 [string range $type 0 [expr $index1 - 1]]
	    set index2 [string first ">" $type]
	    if {$index2<0} {
		return ""
	    } else {
		set name2 [string range $type [expr $index1 +1] \
			                      [expr $index2 - 1]]
	    }
	}

#  	if { $name2 == "unsigned char" || $name2 == "short" || $name2 == "int" || $name2 == "float" || $name2 == "double" }  {
#              return [list "${name1}<unsigned char>" \
#  		         "${name1}<short>" \
#  		         "${name1}<int>" \
#  		         "${name1}<float>" \
#  		         "${name1}<double>" ]
#  	} else {
#  	    return [list $type]
#  	}

        set tmp [list "${name1}<unsigned char> " \
                      "${name1}<short> " \
                      "${name1}<int> " \
  		      "${name1}<float> " \
  		      "${name1}<double> " \
  		      "${name1}<Vector> " \
  		      "${name1}<Tensor> " ]

        set loc [lsearch -exact $tmp $type]
        if { $loc == -1 } {
            return [concat $tmp [list $type] ]
	} else {
	    return $tmp
	}
    }

    method config_labelpair { win text2 } {
#	$win.l2 configure -text $text2
    }

    method config_labeloption {win text2 text3} {
	if {![winfo exists $win]} {
	    return
	}
	$win.l2 configure -text $text2 -val $text2
	$win.l3 configure -text $text3 -val $text3
    }

    method config_labelentry { win text2 } {
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


