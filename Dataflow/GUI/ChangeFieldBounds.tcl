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

itcl_class SCIRun_Fields_ChangeFieldBounds {
    inherit Module
    constructor {config} {
        set name ChangeFieldBounds
        set_defaults
    }

    method set_defaults {} {
	# the width of the first column of the data display
	global $this-firstwidth
	set $this-firstwidth 12

	# these won't be saved 
	global $this-datamin
	global $this-datamax
        global $this-cx
        global $this-cy
        global $this-cz
        global $this-sizex
        global $this-sizey
        global $this-sizez
	set $this-datamin "---"
	set $this-datamax "---"
        set $this-cx "---"
        set $this-cy "---"
        set $this-cz "---"
        set $this-sizex "---"
        set $this-sizey "---"
        set $this-sizez "---"

	# these will be saved
	global $this-datamin2
	global $this-datamax2
	global $this-cdataminmax
        global $this-cx2
        global $this-cy2
        global $this-cz2
        global $this-sizex2
        global $this-sizey2
        global $this-sizez2
	set $this-datamin2 0
	set $this-datamax2 0
	set $this-cdataminmax 0
        set $this-cx2 0
        set $this-cy2 0
        set $this-cz2 0
        set $this-sizex2 0
        set $this-sizey2 0
        set $this-sizez2 0
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }
        toplevel $w

	iwidgets::Labeledframe $w.att -labelpos nw \
		               -labeltext "Input Field Attributes" 
			       
	pack $w.att 
	set att [$w.att childsite]
	
        labelpairmulti $att.l1 "Center (x,y,z)" "[set $this-cx], \
                               [set $this-cy], [set $this-cz]"
        labelpairmulti $att.l2 "Size (x,y,z)" "[set $this-sizex], \
                               [set $this-sizey], [set $this-sizez]" 
	labelpairmulti $att.l3 "Data min,max" "[set $this-datamin], \
		                          [set $this-datamax]"
	pack $att.l1 $att.l2 $att.l3 -side top 

	iwidgets::Labeledframe $w.edit -labelpos nw \
		               -labeltext "Output Field Attributes" 
	pack $w.edit 
	set edit [$w.edit childsite]
	
        labelentry3 $edit.l1 "Center (x,y,z)" $this-cx2 $this-cy2 $this-cz2 \
                    "$this-c update_widget"
        labelentry3 $edit.l2 "Size (x,y,z)" $this-sizex2 $this-sizey2 \
                    $this-sizez2 "$this-c update_widget"
	labelentry2 $edit.l3 "Data min,max" $this-datamin2 $this-datamax2 \
		    $this-cdataminmax

	pack $edit.l1 $edit.l2 $edit.l3 -side top 

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
	$att.l1.l2 configure -text "[set $this-cx], [set $this-cy], \
		                  [set $this-cz]"
	$att.l2.l2 configure -text "[set $this-sizex], [set $this-sizey], \
		                  [set $this-sizez]"
	$att.l3.l2 configure -text "[set $this-datamin], [set $this-datamax]"
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

    method labelentry3 { win text1 text2 text3 text4 func} {
	frame $win 
	pack $win -side top -padx 5
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
	pack $win.l1 $win.colon -side left
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

        if {"[set $this-cdataminmax]"!="1"} {
	  set $this-datamin2 [set $this-datamin]
	  set $this-datamax2 [set $this-datamax]
        }

	set $this-cx2 [set $this-cx]
	set $this-cy2 [set $this-cy]
	set $this-cz2 [set $this-cz]
	set $this-sizex2 [set $this-sizex]
	set $this-sizey2 [set $this-sizey]
	set $this-sizez2 [set $this-sizez]
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




