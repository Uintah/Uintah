
#
# Utility functions
#

#
# To link two variables, so that when one changes the other changes, too
# (update_vars is just the helper function, link_vars is the one to use
#

proc update_vars {v2 v1 vtmp op} {
    global $v1
    global $v2
    if {[set $v1] != [set $v2]} {
	set $v2 [set $v1]
    }
}

proc link_vars {v1 v2} {
    global $v1
    global $v2
    trace variable $v1 w "update_vars $v2"
    trace variable $v2 w "update_vars $v1"
}


#
# Convenience routine for making radiobuttons
#

proc make_labeled_radio {root labeltext command packside variable list} {
    frame $root
    label $root.radiolabel -text $labeltext
    pack $root.radiolabel -padx 2 -side $packside
    set i 0
    foreach t $list {
	set name [lindex $t 0]
	set value $name
	if {[llength $t] == 2} {
	    set value [lindex $t 1]
	}
	radiobutton $root.$i -text "$name" -variable $variable \
		-value $value -command $command
	pack $root.$i -side $packside -anchor nw
	incr i
    }
}

proc change_radio_var {root newvar} {
    foreach t [winfo children $root] {
	set name [winfo name $t]
	if {$name != "radiolabel"} {
	    $t config -variable $newvar
	}
    }
}


#
# Exponential scale
#
itcl_class expscale {
    constructor {config} {
	#
	#  Create a window with the same name as this object
	#
	set class [$this info class]
	::rename $this $this-tmp-
	::frame $this -class $class -borderwidth 2 -relief groove
	::rename $this $this-win-
	::rename $this-tmp- $this

	if {[catch {set variable}]} {
	    set variable $this-variable
	}
	global $variable
	if {[catch {set $variable}]} {
	    set $variable 1
	}

	set built 1
	scale $this.scale -label $label -orient $orient \
		-from -1000000 -to 10000000 -resolution 0.000001 \
		-variable $variable -command $command
	pack $this.scale -fill x -side left -expand yes

	frame $this.e
	pack $this.e -fill y -anchor se

	label $this.e.e -text "E"
	pack $this.e.e -side left

	button $this.e.upexp -text "+" -pady 0 -command "$this upexp"
	button $this.e.downexp -text "-" -pady 0 -command "$this downexp"
	pack $this.e.upexp -side top -fill x
	pack $this.e.downexp -side bottom -fill x

	entry $this.e.exp -width 3 -selectborderwidth 0 -highlightthickness 0 \
		-insertborderwidth 0
	pack $this.e.exp

	setscales
    }
    destructor {
	destroy $this.scale
    }
    method config {config} {
    }
    public label "" 
    public orient
    protected built 0
    public variable "" {
	if {$built} {
	    $this.scale config -variable ""
	    setscales
	    $this.scale config -variable $variable
	}
    }
    protected exp 0
    public command ""

    method upexp {} {
	incr exp
	setscales
    }
    method downexp {} {
	if {$exp < -4} {
	    return;
	}
	incr exp -1
	setscales
    }
    method setscales {} {
	global $variable
	puts "variable is $variable"
	set value [set $variable]
	set mag [expr pow(10, $exp)]
	puts "mag is $mag"
	# Round the value down to get from...
	puts "value is $value"
	set from [expr int($value/$mag)*$mag]
	puts "from is $from"

	set to [expr $from+$mag]
	puts "to is $to"
	set ti [expr $mag/5]
	$this.scale configure -from $from -to $to -tickinterval $mag

	$this.e.exp delete 0 end
	$this.e.exp insert 0 $exp
    }
}
