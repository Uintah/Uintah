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
	global $this-numelems2
	global $this-dataat2
	global $this-minx2
	global $this-miny2
	global $this-minz2
	global $this-maxx2
	global $this-maxy2
	global $this-maxz2
	global $this-cfldname
	global $this-ctypename
	global $this-cdataminmax
	global $this-cbboxmin
	global $this-cbboxmax
	global $this-cnumelems
	global $this-cdataat
	set $this-fldname2 ""
	set $this-typename2 ""
	set $this-datamin2 ""
	set $this-datamax2 ""
	set $this-numelems2 ""
	set $this-dataat2 ""
	set $this-minx2 ""
	set $this-miny2 ""
	set $this-minz2 ""
	set $this-maxx2 ""
	set $this-maxy2 ""
	set $this-maxz2 ""
	set $this-cfldname 0
	set $this-ctypename 0
	set $this-cdataminmax 0
	set $this-cbboxmin 0
	set $this-cbboxmax 0
	set $this-cnumelems 0
	set $this-cdataat 0
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
	
	labelpair $att.l1 "Name" "[set $this-fldname]"
	labelpair $att.l2 "Typename" "[set $this-typename]"
	labelpair $att.l3 "BBox min" "[set $this-minx], [set $this-miny], \
		                      [set $this-minz]"
	labelpair $att.l4 "BBox max" "[set $this-maxx], [set $this-maxy], \
		                      [set $this-maxz]"
	labelpair $att.l5 "Data min,max" "[set $this-datamin], \
		                          [set $this-datamax]"
	labelpair $att.l7 "# Nodes" "[set $this-numnodes]"
	labelpair $att.l8 "# Elements" "[set $this-numelems]"
	labelpair $att.l9 "Data at" "[set $this-dataat]"
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
	labelentry3 $edit.l3 "BBox min" $this-minx2 $this-miny2 $this-minz2 \
		    $this-cbboxmin
	labelentry3 $edit.l4 "BBox max" $this-maxx2 $this-maxy2 $this-maxz2 \
                    $this-cbboxmax
	labelentry2 $edit.l5 "Data min,max" $this-datamin2 $this-datamax2 \
		    $this-cdataminmax
	labeloption $edit.l8 "# Elements" "[set $this-numelems2]" "0" \
		    $this-numelems2 $this-cnumelems 
	labelcombo $edit.l9 "Data at" {Field::CELL Field::FACE Field::EDGE \
		                       Field::NODE Field::NONE} \
		   $this-dataat2 $this-cdataat
	pack $edit.l1 $edit.l2 $edit.l3 $edit.l4 $edit.l5 \
	     $edit.l8 $edit.l9 -side top 
    }

    method labelpair { win text1 text2 } {
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

    method labelentry3 { win text1 text2 text3 text4 var } {
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
    }

    method labelcombo { win text1 arglist var var2} {
	frame $win 
	pack $win -side top -padx 5
	checkbutton $win.b -var $var2
	label $win.l1 -text $text1 -width [set $this-firstwidth] \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left
	iwidgets::optionmenu $win.c -foreground darkred 
	set i 0
	set length [llength $arglist]
	for {set elem [lindex $arglist $i]} {$i<$length} \
	    {incr i 1; set elem [lindex $arglist $i]} {
	    $win.c insert end $elem
	}
	label $win.l2 -text "" -width 40 -anchor w -just left
	pack $win.b $win.l1 $win.colon -side left
	pack $win.c $win.l2 -side left	
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

    method update_attributes {} {
	set w .ui[modname]
	set att [$w.att childsite]
	set edit [$w.edit childsite]

	config_labelpair $att.l1 "[set $this-fldname]"
	config_labelpair $att.l2 "[set $this-typename]"
	config_labelpair $att.l3 "[set $this-minx], [set $this-miny], \
                                  [set $this-minz]"
	config_labelpair $att.l4 "[set $this-maxx], [set $this-maxy], \
                                  [set $this-maxz]"
	config_labelpair $att.l5 "[set $this-datamin], [set $this-datamax]"
	config_labelpair $att.l7 "[set $this-numnodes]"
	config_labelpair $att.l8 "[set $this-numelems]"
	config_labelpair $att.l9 "[set $this-dataat]"
    }

    method copy_attributes {} {
	set w .ui[modname]
	set att [$w.att childsite]
	set edit [$w.edit childsite]
	if {"[set $this-fldname]"!="--- Name Not Assigned ---"} {
	    set $this-fldname2 [set $this-fldname]
	} else {
	    set $this-fldname2 ""
	}
	config_labeloption $edit.l8 "[set $this-numelems]" "0"
	set $this-numelems2 [set $this-numelems]
	config_labelcombo $edit.l9 {Field::CELL Field::FACE Field::EDGE \
		                    Field::NODE Field::NONE} [set $this-dataat]
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

	if {"$name1"=="LatticeVol"} {
	    if {"$name2"!="Vector" && "$name2"!="Tensor"} {
		return { LatticeVol<char> "LatticeVol<unsigned char>" \
                         LatticeVol<short> "LatticeVol<unsigned short>" \
			 LatticeVol<int> "LatticeVol<unsigned int>" \
			 LatticeVol<float> LatticeVol<double> }
	    }
	} elseif {"$name1"=="TetVol"} {
	    if {"$name2"!="Vector" && "$name2"!="Tensor"} {
		return { TetVol<char> "TetVol<unsigned char>" \
                         TetVol<short> "TetVol<unsigned short>" \
			 TetVol<int> "TetVol<unsigned int>" \
			 TetVol<float> TetVol<double> }
	    }
	} elseif {"$name1"=="TriSurf"} {
	    if {"$name2"!="Vector" && "$name2"!="Tensor"} {
		return { TriSurf<char> "TriSurf<unsigned char>" \
                         TriSurf<short> "TriSurf<unsigned short>" \
			 TriSurf<int> "TriSurf<unsigned int>" \
			 TriSurf<float> TriSurf<double> }
	    }
	} else {
	    return ""
	}
    }

    method config_labelpair { win text2 } {
	$win.l2 configure -text $text2
    }

    method config_labeloption {win text2 text3} {
	$win.l2 configure -text $text2 -val $text2
	$win.l3 configure -text $text3 -val $text3
    }

    method config_labelentry { win text2 } {
    }

    method config_labelcombo { win arglist sel} {
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


