itcl_class ModelCreation_Converter_MatrixToField {
    inherit Module
    constructor {config} {
        set name MatrixToField
        set_defaults
    }

    method set_defaults {} {
        global $this-datalocation
        
        set $this-datalocation "Node"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w


        labelcombo $w.datalocation "Data Location" \
          {"Node" "Element"} \
          $this-datalocation


        makeSciButtonPanel $w $w $this
        moveToCursor $w
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

	$win.c select [set $var]

	label $win.l2 -text "" -width 20 -anchor w -just left

	# hack to associate optionmenus with a textvariable
	# bind $win.c <Map> "$win.c select {[set $var]}"

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

    method set_combobox { win var name1 name2 op } {
	set w .ui[modname]
	set menu $w.$win
	if {[winfo exists $menu]} {
	    $menu select $var
	}
    }
}


