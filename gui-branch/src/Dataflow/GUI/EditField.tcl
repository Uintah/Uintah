itcl::class SCIRun_Fields_EditField {
    inherit ModuleGui

    # the width of the first column of the data display
    public variable firstwidth 12
    
    # these won't be saved 
    public variable fldname "---"
    public variable typename "---"
    public variable datamin "---"
    public variable datamax "---"
    public variable numnodes "---"
    public variable numelems "---"
    public variable dataat "---"
    public variable cx "---"
    public variable cy "---"
    public variable cz "---"
    public variable sizex "---"
    public variable sizey "---"
    public variable sizez "---"
    
    # these will be saved
    public variable fldname2 ""
    public variable typename2 "--- No typename ---"
    public variable datamin2 0
    public variable datamax2 0
    public variable dataat2 "Nodes"
    public variable cfldname 0
    public variable ctypename 0
    public variable cdataminmax 0
    public variable cgeom 0
    public variable cnumelems 0
    public variable cdataat 0
    public variable cx2 0
    public variable cy2 0
    public variable cz2 0
    public variable sizex2 0
    public variable sizey2 0
    public variable sizez2 0
    
    constructor {} {
        set name EditField
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
	
	labelpair $att.l1 "Name" fldname
	labelpair $att.l2 "Typename" typename
        labelpairmulti $att.l3 "Center (x,y,z)" "$cx, $cy, $cz"
        labelpairmulti $att.l4 "Size (x,y,z)" "$sizex, $sizey $sizez" 
	labelpairmulti $att.l5 "Data min,max" "$datamin, \
		                          $datamax"
	labelpair $att.l7 "# Nodes" numnodes
	labelpair $att.l8 "# Elements" numelems
	labelpair $att.l9 "Data at" dataat
	pack $att.l1 $att.l2 $att.l3 $att.l4 $att.l5 \
	     $att.l7 $att.l8 $att.l9 -side top 

	button $w.copy -text "copy input to output" \
                       -command "$this copy_attributes"
	pack $w.copy -side top -pady 5

	iwidgets::Labeledframe $w.edit -labelpos nw \
		               -labeltext "Output Field Attributes" 
			       
	pack $w.edit 
	set edit [$w.edit childsite]
	
	labelentry $edit.l1 "Name" fldname2 cfldname
	labelcombo $edit.l2 "Typename" \
		   "[possible_typenames $typename2]" \
                   typename2 ctypename
        labelentry3 $edit.l3 "Center (x,y,z)" cx2 cy2 cz2 \
                    cgeom "$this-c update_widget"
        labelentry3 $edit.l4 "Size (x,y,z)" sizex2 sizey2 \
                    sizez2 cgeom "$this-c update_widget"
	labelentry2 $edit.l5 "Data min,max" datamin2 datamax2 \
		    cdataminmax
	labelcombo $edit.l9 "Data at" {Nodes Edges Faces Cells} \
		   dataat2 cdataat
	pack $edit.l1 $edit.l2 $edit.l5 \
	     $edit.l9 -side top 


	frame $w.exec
	pack $w.exec -side bottom -padx 5 -pady 5
	button $w.exec.execute -text "Execute" -command "$this-c needexecute"
	pack $w.exec.execute -side top -e n
    }

    method update_multifields {} {
        set w .ui[modname]
	if {![winfo exists $w]} {
	    return
	}
	set att [$w.att childsite]
	$att.l3.l2 configure -text "$cx, $cy, $cz"
	$att.l4.l2 configure -text "$sizex, $sizey, $sizez"
	$att.l5.l2 configure -text "$datamin, $datamax"
    }

    method labelpair { win text1 var } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width $firstwidth \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -textvar [scope $var] -width 40 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon $win.l2 -side left
    } 

    method labelpairmulti { win text1 text2 } {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 -width $firstwidth \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	label $win.l2 -text $text2 -width 40 -anchor w -just left \
		-fore darkred
	pack $win.l1 $win.colon $win.l2 -side left
    } 

    method labelentry { win text1 text2 var } {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var [scope $var]
	label $win.l1 -text $text1 -width $firstwidth \
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
	checkbutton $win.b -var [scope $var]
	label $win.l1 -text $text1 -width $firstwidth \
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
	checkbutton $win.b -var [scope $var]
	label $win.l1 -text $text1 -width $firstwidth \
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
	checkbutton $win.b -var [scope $var2]
	label $win.l1 -text $text1 -width $firstwidth \
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
	checkbutton $win.b -var [scope $var2]
	label $win.l1 -text $text1 -width $firstwidth \
                      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left 
	radiobutton $win.l2 -text $text2 -var [scope $var] -val $text2 -fore darkred
	radiobutton $win.l3 -text $text3 -var [scope $var] -val $text3 -fore darkred
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
        if {"$cfldname"!="1"} {
	  if {"$fldname"!="--- Name Not Assigned ---"} {
	    set fldname2 $fldname
	  } else {
	    set fldname2 ""
	  }
        }
        if {"$cdataat"!="1"} {
	  config_labelcombo $edit.l9 {Nodes Edges Faces Cells} \
            $dataat
        }
        if {"$cdataminmax"!="1"} {
	  set datamin2 $datamin
	  set datamax2 $datamax
        }
        if {"$cgeom"!="1"} {
          set cx2 $cx
          set cy2 $cy
          set cz2 $cz
          set sizex2 $sizex
          set sizey2 $sizey
          set sizez2 $sizez
        }
        if {"$typename"!="1"} {
	  config_labelcombo $edit.l2 [possible_typenames \
            $typename] $typename
        }
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




