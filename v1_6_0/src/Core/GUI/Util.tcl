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

# Utility functions

# To link two variables, so that when one changes the other changes, too
# (update_vars is just the helper function, link_vars is the one to use

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


# Convenience routine for making radiobuttons

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


# Exponential scale
itcl_class expscale {

    protected decimalplaces

    method modname { c } {
	set n [string range $c [expr [string last "::" $c] + 2] end]
	return $n
    }

    constructor {config} {

	#  Create a window with the same name as this object

	#puts "entering expscale constructor"
	#puts "dollar this is $this"
	#puts "name received by expscale = [modname $this]"
	#puts "dollar this info class = [ $this info class] "

	set decimalplaces 3

	set class [modname [$this info class]]
	if {[catch {set w}]} {
	    set w [modname $this]
	}

	frame $w -class $class -borderwidth 2 -relief groove

	# This tells the expscale to delete itself when its
	# window is destroyed.  That way, when the expscale
	# is "re-created" there isn't a name conflict.
	# (A better solution is to not let the GUI windows
	# be destroyed in the first place... just close 
	# (unmap) them and remap them when they need to
	# be seen again.
	bind $w <Destroy> "::delete object $this"

	pack $w -side top -fill both -expand 1
	if {[catch {set variable}]} {
	    set variable $w-variable
	}
	global $variable
	if {[catch {set $variable}]} {
	    set $variable 1
	}

	set tmpvar [set $variable]

	set resolution [getResolution ]
	set built 1
	scale $w.scale -label $label -orient $orient \
		-from -1000000 -to 10000000 -resolution $resolution \
		-variable $variable -command $command
	pack $w.scale -fill x -side left -expand yes

	frame $w.e
	pack $w.e -fill y -anchor se

	label $w.e.e -text "E"
	pack $w.e.e -side left

	button $w.e.upexp -text "+" -pady 0 -command "$this upexp"
	button $w.e.signchange -text "S" -pady 0 -command "$this chngsign"
	button $w.e.downexp -text "-" -pady 0 -command "$this downexp"
	pack $w.e.upexp -side top -fill x
	pack $w.e.signchange -side bottom -fill x
	pack $w.e.downexp -side bottom -fill x

	entry $w.e.exp -width 3 -selectborderwidth 0 -highlightthickness 0 \
		-insertborderwidth 0
	pack $w.e.exp

	newvalue $tmpvar
    }

    destructor {
	destroy $w.scale
	destroy $w.e
    }

    method config {config} {
    }

    public label "" 
    public orient ""
    public w
    protected built 0
    public variable "" {
	if {$built} {
	    $w.scale config -variable ""
	    updatevalue [set $variable]
	    $w.scale config -variable $variable
	}
    }

    method getResolution {} {
	if { $decimalplaces == 0 } {
	    return 1
	} else {
	    set val 1

	    for { set i 1} { $i < $decimalplaces } {incr i } {
		set val "0$val"
	    }
	    
	    return "0.$val"
	}
    }
    protected exp 0
    protected sign 1
    public command ""

    method upexp {} {
	incr exp
	if { $exp > 2 } {
	    set decimalplaces 0
	} else {
	    incr decimalplaces -1
	}
	global $variable
	updatevalue [set $variable]
    }

    method downexp {} {
	if {$exp < -5} {
	    return;
	}
	incr exp -1
	if { $exp > 2 } {
	    set decimalplaces 0
	} else {
	    incr decimalplaces
	}
	global $variable
	updatevalue [set $variable]
    }

    method chngsign {} {
	incr sign [expr $sign * -2]
	global $variable
	updatevalue [expr -1 * [set $variable]]
	if {[set $variable] == 0} {
	    updatevalue -0.00000000001
	}
    }

    # don't change the number of decimalplaces or the exponent - just move
    #   the variable and set the to-from bounds on the scale

    method updatevalue {value} {
	set mag [expr pow(10, $exp)]
	# Round the value down to get from...
	if {$value < 0} {
	    set from [expr int($value/$mag-1)*$mag]
	} else {
	    set from [expr int($value/$mag)*$mag]
	}

	set to [expr $from+$mag]
	set ti [expr $mag/5]
	
	set resolution [getResolution ]
	$w.scale configure -from $from -to $to \
	    -tickinterval $mag -resolution $resolution

	global $variable
	set $variable $value
	$w.e.exp delete 0 end
	$w.e.exp insert 0 $exp
    }
	
    # figure out everything - decimalplaces, exponent, and to-from bounds

    method newvalue {value} {
	global $variable
	
	# crop of trailing zeros
	set value [expr $value * 1]
	set decimalplaces 2
	set spl [split $value .]
	if {[llength $spl] != 1} {
	    set decimalplaces [string length [lindex $spl end]]
	}
	if {$decimalplaces > 8} {
	    set decimalplaces 8
	}

	set exp 2
	if {$value != 0} {
	    while {([expr pow(10, [expr $exp-1])] > [expr abs($value)]) && \
		    ($exp >= -5)} {
		
		incr exp -1
	    }
	}
	set mag [expr pow(10, $exp)]
	if {$value < 0} {
	    set sign -1
	}
	# Round the value down to get from...
	if {$value < 0} {
	    set from [expr int($value/$mag-1)*$mag]
	} else {
	    set from [expr int($value/$mag)*$mag]
	}

	set to [expr $from+$mag]
	set ti [expr $mag/5]
	
	set resolution [getResolution ]

	$w.scale configure -from $from -to $to \
	    -tickinterval $mag -resolution $resolution
	set $variable $value

	$w.e.exp delete 0 end
	$w.e.exp insert 0 $exp
    }	
}
