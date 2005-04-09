itcl_class BuildMultiMesh {
    inherit Module
    constructor {config} {
	set name BuildMultiMesh
	set_defaults
    }
    method set_defaults {} {
	global $this-numSources
	global $this-levels
	global $this-gl_falloff
	global $this-min_weight
	global $this-max_weight	
	set $this-numSources 0
	set $this-levels 4
	set $this-gl_falloff 50
	set $this-min_weight 1
	set $this-max_weight 10	
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	wm minsize $w 300 100
	frame $w.f -width 400 -height 500
	pack $w.f -padx 2 -pady 2 -side top -fill x -expand yes
	set n "$this-c needexecute"
	set p "$this try_partial_execute"
	frame $w.f.info
	pack $w.f.info -side top
	global $this-levels
	global $this-gl_falloff
	scale $w.f.info.levels -from 1 -to 10 -label "Number of Decimation Levels: " \
		-showvalue true -tickinterval 1 -variable $this-levels \
		-orient horizontal -length 250
	trace variable $this-levels w $p
	scale $w.f.info.percent -from 1 -to 99 -label "Node Global Falloff Rate (%): "\
		-showvalue true -tickinterval 14 -variable $this-gl_falloff \
		-orient horizontal -length 250
	trace variable $this-gl_falloff w $p
	pack $w.f.info.levels $w.f.info.percent -side top
	set i 0
	global $this-numSources
	while {$i<[set $this-numSources]} {
	    incr i
	    frame $w.f.s$i -relief groove -borderwidth 2
	    pack $w.f.s$i -side top -fill x -expand 1
	    global $this-s$i
	    set $this-s$i 1
	    checkbutton $w.f.s$i.label -text "Source $i" -variable $this-s$i \
		    -command "$this do_source $i"
	    $w.f.s$i.label select
	    trace variable $this-s$i w $p
	    frame $w.f.s$i.scales
	    pack $w.f.s$i.label $w.f.s$i.scales -side top -fill x -expand 1
	    global $this-ch$i
	    set $this-ch$i 1
            scale $w.f.s$i.scales.charge -from -10 -to 10 -label "Charge:" \
		    -showvalue true -tickinterval 2.5 -variable $this-ch$i \
		    -orient horizontal -length 250 -resolution .1
	    $w.f.s$i.scales.charge set 1
	    trace variable $this-ch$i w $p
	    global $this-fa$i
	    set $this-fa$i 1.0
	    scale $w.f.s$i.scales.falloff -from .01 -to 4 -label "Falloff (1/d^n): "\
		    -showvalue true -tickinterval .5 -variable $this-fa$i \
		    -orient horizontal -length 250 -resolution .01
	    $w.f.s$i.scales.falloff set 1.0
	    trace variable $this-fa$i w $p
	    global $this-sz$i
	    set $this-sz$i .2
	    scale $w.f.s$i.scales.size -from .01 -to 4 -label "Size:" \
		    -showvalue true -tickinterval 2.5 -variable $this-sz$i \
		    -orient horizontal -length 250 -resolution .01
	    $w.f.s$i.scales.size set .2
	    trace variable $this-sz$i w $p

	    pack $w.f.s$i.scales.charge $w.f.s$i.scales.falloff \
		    $w.f.s$i.scales.size -side top -fill x -expand 1
	}
	frame $w.b
	pack $w.b -side bottom
	button $w.b.print_values -text "Print Values" -relief raised \
		-command "$this print_values"
	checkbutton $w.b.same -text "Same Inputs?" -variable $this-sameInput
	$w.b.same select

	global $this-PE
	set $this-PE 0
	checkbutton $w.b.partial -text "Partial Execute" -variable $this-PE
	$w.b.partial deselect
	trace variable $this-PE w "$this change_pe"

	button $w.b.go -text "Execute" -relief raised \
		-command $n
	pack $w.b.print_values $w.b.same $w.b.partial $w.b.go -side top
    }

    method change_pe {n1 n2 op} {
	global $n1
	if {[set $n1]} {
	    $this-c partial_execute
	}
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
    method try_partial_execute {n1 n2 op} {
	global $this-PE
	if {[set $this-PE]} {
	    $this-c partial_execute
	}
    }
}
