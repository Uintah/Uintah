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

itcl_class SCIRun_Fields_ChangeFieldDataType {
    inherit Module
    constructor {config} {
        set name ChangeFieldDataType
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	# these won't be saved 
	global $this-fldname
	global $this-inputtypename
	set $this-fldname "---"
	set $this-inputtypename "---"

	# these will be saved
	global $this-outputtypename
	set $this-outputtypename "--- No typename ---"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	iwidgets::Labeledframe $w.input -labelpos nw \
		               -labeltext "Input Field Type" 
			       
	pack $w.input 
	set input [$w.input childsite]
	
	labelpair $input.name "Name" $this-fldname
	labelpair $input.typename "Typename" $this-inputtypename
	pack $input.name $input.typename -side top 

	iwidgets::Labeledframe $w.output -labelpos nw \
		               -labeltext "Output Field Type" 
			       
	pack $w.output 
	set output [$w.output childsite]
	
	labelcombo $output.name "Typename" \
		   "[possible_typenames [set $this-outputtypename]]" \
                   $this-outputtypename
	pack $output.name -side top 

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

    method labelcombo { win text1 arglist var } {
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
	set output [$w.output childsite]

        if {"[set $this-inputtypename]"!="1"} {
	  config_labelcombo $output.name [possible_typenames \
            [set $this-inputtypename]] [set $this-inputtypename]
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
	global $this-outputtypename
	set oldtype [set $this-outputtypename]
	$win.c delete 0 end
	if {[llength $arglist]==0} {
	    $win.c insert end ""
	}
	set i 0
	set found 0
	set length [llength $arglist]
	for {set elem [lindex $arglist $i]} {$i<$length} \
	    {incr i 1; set elem [lindex $arglist $i]} {
		if {$elem == $oldtype} {
		    set found 1; 
		}
		$win.c insert end $elem
	}
    
	if {$found == 1} {
	    $win.c select $oldtype
	} else {
	    if {"$sel"!="---"} {
		$win.c select $sel
	    }
	}
    }
}
