itcl_class ModelCreation_FieldsData_FieldDataElemToNode {
    inherit Module
    constructor {config} {
        set name FieldDataElemToNode
        set_defaults
    }

    method set_defaults {} {
    	global $this-method
      set $this-method "Interpolate"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w

        iwidgets::Labeledframe $w.method -labelpos nw \
		               -labeltext "Method for Computing Node Data" 
			       
        pack $w.method 
        set method [$w.method childsite]
        labelcombo $method.method "Method" \
            {Interpolate Average Min Max Sum Median None} \
            $this-method $this-temp
        pack $method.method -side top 

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
    
    method labelcombo { win text1 arglist var var2} {
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


