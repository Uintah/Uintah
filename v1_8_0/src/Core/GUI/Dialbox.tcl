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

#  Dialbox.tcl
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   May 1995
#  Copyright (C) 1995 SCI Group

catch {rename Dialbox ""}

# The legal dialtypes are:
# (Use (dialtype)_dial to create a dial.)
# bounded:     between min and max, inclusive
# unbounded:   infinite crank mode
# wrapped:     between min and max, but goes from min to max
#              and vice versa (e.g. 0-360 degree angle)
#              (min IS max)
# unconnected: This is the initial state of all dials.

itcl_class Dialbox {
    constructor {titletext config} {
	set title "$titletext"
	for {set i 0} {$i < 8} {incr i 1} {
	    set dialnames($i) "Unconnected"
	    set dialtypes($i) "Unconnected"
	    set dialvals($i) 0.0
	    set dialmins($i) 0.0
	    set dialmaxs($i) 100.0
	    set dialscales($i) 1.0
	    set dialcommands($i) ""
	    set dialstopcommands($i) ""
	}
    }

    destructor {
	set tc $connections
	set idx [lsearch -exact $tc $this]
	if {$idx != -1} {
	    if {$idx == 0} {
		if {[llength $tc] == 1} {
		    puts "Need to disconnect dialbox!!"
		} else {
		    set connections [lreplace $tc $idx $idx]
		    eval [lindex $tc 1] connect
		}
	    } else {
		set connections [lreplace $tc $idx $idx]
	    }
	}
    }

    method gettitle {} {
	return "$title"
    }
    
    method storeval {dial val} {
	set dialvals($dial) $val
    }

    protected title ""
    protected dialtypes
    protected dialnames
    protected dialvals
    protected dialmins
    protected dialmaxs
    protected dialscales
    protected dialcommands
    protected dialstopcommands

    method bounded_dial {dial name val min max scale command {stopcmd ""}} {
	if {$dial < 0 || $dial > 7} {
	    puts "Invalid bounded_dial index '$dial' for '$title'."
	    return
	}
	set dialtypes($dial) "bounded"
	set dialmins($dial) $min
	set dialmaxs($dial) $max
	standard_dial $dial $name $val $scale $command $stopcmd
    }

    method unbounded_dial {dial name val scale command {stopcmd ""}} {
	if {$dial < 0 || $dial > 7} {
	    puts "Invalid unbounded_dial index '$dial' for '$title'."
	    return
	}
	set dialtypes($dial) "unbounded"
	standard_dial $dial $name $val $scale $command $stopcmd
    }
    
    

    method wrapped_dial {dial name val min max scale command {stopcmd ""}} {
	if {$dial < 0 || $dial > 7} {
	    puts "Invalid wrapped_dial index '$dial' for '$title'."
	    return
	}
	set dialtypes($dial) "wrapped"
	set dialmins($dial) $min
	set dialmaxs($dial) $max
	standard_dial $dial $name $val $scale $command $stopcmd
    }

    # Used by (dialtype)_dial methods only.
    method standard_dial {dial name val scale command {stopcmd ""}} {
	set dialnames($dial) "$name"
	set dialvals($dial) $val
	set dialscales($dial) $scale
	set dialcommands($dial) "$command"
	set dialstopcommands($dial) "$stopcmd"
    }

    protected after_id ""

    method dial_moved {dial val} {
	if {$after_id != ""} {
	    after cancel $after_id
	}
	set after_id [after 500 $this dial_stopped $dial $val]
	if {$dialcommands($dial) != ""} {
	    eval $dialcommands($dial) $val
	}
    }

    method dial_stopped {dial val} {
	puts "dial_stopped called..."
	set after_id ""
	if {$dialstopcommands($dial) != ""} {
	    puts "stop command is $dialstopcommands($dial)"
	    eval $dialstopcommands($dial) $val
	}
    }

    method dial_scale {dial scale} {
	if {$dial < 0 || $dial > 7} {
	    puts "Invalid dial_scale index '$dial' for '$title'."
	    return
	}
	set dialscales($dial) $scale
	global dialbox-dialscale$dial
	set dialbox-dialscale$dial $dialscales($dial)
    }

    method connect {} {
	global dialbox-title
	set dialbox-title "$title"

	if {[llength $connections] != 0} {
	    set connected [lindex $connections 0]
	    for {set i 0} {$i < 8} {incr i 1} {
		global dialbox-dialval$i
		$connected storeval $i [set dialbox-dialval$i]
	    }
	}

	for {set i 0} {$i < 8} {incr i 1} {
	    global dialbox-dialtype$i
	    set dialbox-dialtype$i $dialtypes($i)
	    global dialbox-dialname$i
	    set dialbox-dialname$i "$dialnames($i)"
	    global dialbox-dialval$i
	    set dialbox-dialval$i $dialvals($i)
	    global dialbox-dialmin$i
	    set dialbox-dialmin$i $dialmins($i)
	    global dialbox-dialmax$i
	    set dialbox-dialmax$i $dialmaxs($i)
	    global dialbox-dialscale$i
	    set dialbox-dialscale$i $dialscales($i)
	}

	set tc $connections
	set idx [lsearch -exact $tc $this]
	if {$idx != -1} {
	    set tc [lreplace $tc $idx $idx]
	}
	set tc [concat [list $this] $tc]
	if {[llength $tc] > 10} {
	    set tc [lreplace $tc 10 end]
	}
	set connections $tc

	ui
    }

    proc ui {} {
	set w .uidialbox
	set dialbox $w.f
	if {[winfo exists $w]} {
	    $dialbox.connections.menu delete 0 last
	    for {set i 0} {$i < [llength $connections]} {incr i 1} {
		set newtitle "[[lindex $connections $i] gettitle]"
		
		$dialbox.connections.menu add command -label "$newtitle" \
			-command "Dialbox :: makeconnection $i"
	    }
	    for {set i 0} {$i < 8} {incr i 1} {
		global dialbox-dial$i
		[set dialbox-dial$i] resetPos
	    }
	    raise $w
	    return
	}

	toplevel $w
	wm minsize $w 100 100
	frame $w.f
	set size 100

	menubutton $dialbox.connections -textvariable dialbox-title \
		-menu $dialbox.connections.menu \
		-relief raised -borderwidth 4
	menu $dialbox.connections.menu -tearoff no
	for {set i 0} {$i < [llength $connections]} {incr i 1} {
	    set newtitle "[[lindex $connections $i] gettitle]"
	    $dialbox.connections.menu add command -label "$newtitle" \
		    -command "Dialbox :: makeconnection $i"
	}

	for {set i 0} {$i < 8} {incr i 2} {
	    set ii [expr $i+1]
	    frame $dialbox.dials$i$ii
	    for {set j $i} {$j <= $ii} {incr j 1} {
		DialW $dialbox.dials$i$ii.dial$j -initSize $size \
			-dialtypeVariable dialbox-dialtype$j \
			-textVariable dialbox-dialname$j \
			-variable dialbox-dialval$j \
			-minVariable dialbox-dialmin$j \
			-maxVariable dialbox-dialmax$j \
			-scaleVariable dialbox-dialscale$j \
			-command "Dialbox :: notifydialbox $j"
		global dialbox-dial$j
		set dialbox-dial$j $dialbox.dials$i$ii.dial$j
	    }
	    pack $dialbox.dials$i$ii.dial$i $dialbox.dials$i$ii.dial$ii \
		    -side left -fill both -expand yes
	}

	pack $dialbox.connections $dialbox.dials01 $dialbox.dials23 \
		$dialbox.dials45 $dialbox.dials67 \
		-side top -fill both -expand yes
	pack $dialbox -fill both -expand yes
    }

    proc notifydialbox {dial val} {
	set connected [lindex $connections 0]
	if {$connected != ""} {
	    $connected dial_moved $dial $val
	}
    }

    proc makeconnection {which} {
	[lindex $connections $which] connect
    }

    common connections {}
}
