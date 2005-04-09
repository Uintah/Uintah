itcl_class MMeshToSFUG {
    inherit Module
    constructor {config} {
	set name MMeshToSFUG
	set_defaults
    }
    method set_defaults {} {
	global $this-total_levels
	global $this-selected_level
	set $this-total_levels 6
	set $this-selected_level 1
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 100
	frame $w.f -width 300 -height 100
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute "
	global $this-total_levels
	scale $w.f.s -from 1 -to 6 -label "Current MultiMesh Level: " \
		-showvalue true -tickinterval 1 \
		-variable $this-selected_level -orient horizontal -length 250 \
		-command $n
	button $w.f.go -text "Execute" -relief raised \
		-command $n
	pack $w.f.s $w.f.go -side top
	global $this-total_levels
	trace variable $this-total_levels w "$this change_total"
    }

    method change_total {n1 n2 op} {
	global $this-total_levels
	set w .ui$this
	global $w.f.s
	$w.f.s configure -to [set $this-total_levels]
    }

    method print_values {} {
	set i 0
	global $this-numSources
	global $this-levels
	global $this-gl_falloff
	puts -nonewline "Number of levels: "
	puts [set $this-levels]
	puts -nonewline "Global Falloff Rate (per level): "
	puts -nonewline [set $this-gl_falloff]
	puts "%"
	while {$i<[set $this-numSources]} {
	    incr i
	    puts -nonewline "Source "
	    puts $i
	    global $this-s${i}ch
	    global $this-s${i}fa
	    global $this-s${i}sz
	    puts -nonewline "  Charge: "
	    puts -nonewline [set $this-s${i}ch]
	    puts -nonewline "  Falloff: 1/dist^"
	    puts -nonewline [set $this-s${i}fa]
	    puts -nonewline "  Size: "
	    puts [set $this-s${i}sz]
	}
	puts " "
    }
    method do_source {i} {
	global $this-s$i
	set w .ui$this
	if {[set $this-s$i]} {
	    $w.f.s$i.scales.charge configure -state normal
	    $w.f.s$i.scales.falloff configure -state normal
	} else {
	    $w.f.s$i.scales.charge configure -state disabled
	    $w.f.s$i.scales.falloff configure -state disabled
	}
    }
    method try_partial_execute {v} {
	global $this-PE
	if {[set $this-PE]} {
	    $this-c partial_execute
	}
    }
    method try_partial_execute_b {} {
	global $this-PE
	if {[set $this-PE]} {
	    $this-c partial_execute
	}
    }
}