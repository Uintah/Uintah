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


proc showMemStats {} {
    global memstats_window
    if [catch {raise $memstats_window}] {
	toplevel .msw
	wm title .msw "Memory Statistics"
	wm iconname .msw "MemStats"
	wm minsize .msw 100 100
	set memstats_window .msw
	canvas .msw.canvas -yscrollcommand ".msw.vscroll set" \
		-scrollregion {0c 0c 11c 50c} \
		-width 11c -height 18c -borderwidth 0
	scrollbar .msw.vscroll -relief sunken -command ".msw.canvas yview"
	pack .msw.vscroll -side right -fill y -padx 4 -pady 4
	pack .msw.canvas -expand yes -fill y -pady 4
	set lineheight [winfo pixels .msw.canvas 8p]
	set gleft [winfo pixels .msw.canvas 8.5c]
	set gwidth [winfo pixels .msw.canvas 2c]
	set font -Adobe-courier-bold-r-*-100-75-*
	set bins [memstats nbins]
        set left 4
	for {set i 0} {$i<$bins} {incr i} {
	    set top [expr ($i+19)*$lineheight]
	    .msw.canvas create text $left [expr $top+1] \
		    -tag t$i -anchor nw -font $font
	    .msw.canvas create text $left $top -tag gt$i -anchor nw -font $font
	    .msw.canvas create rectangle $left 0 0 0 -tag ga$i \
		    -fill blue
	    .msw.canvas create rectangle $left 0 0 0 -tag gb$i \
		    -fill red
	}
	.msw.canvas create text $left 4 -tag glob -anchor nw -font $font
	button .msw.canvas.audit -text "Audit" -command "memstats audit" \
		-borderwidth 3
	.msw.canvas create window 9c 1c -window .msw.canvas.audit
	button .msw.canvas.dump -text "Dump" -command "memstats dump" \
		-borderwidth 3
	.msw.canvas create window 9c 2c -window .msw.canvas.dump
	after 0 updateMemStats $lineheight $gleft $gwidth
    }
}

proc updateMemStats {lineheight gleft gwidth } {
    if {[winfo exists .msw] == 0} {
	return
    }
    set glob [memstats globalstats]
    if {$glob != ""} {
	.msw.canvas itemconfigure glob -text $glob
	foreach i [split [memstats binchange]] {
	    if {$i >= 0} {
		set info [split [memstats bin $i] "|"]
		set line [lindex $info 0]
		set inlist [lindex $info 1]
		set diff [lindex $info 2]
		set text [lindex $info 3]
		.msw.canvas itemconfigure t$line -text $text
		set top [expr ($line+19)*$lineheight]
		set bot [expr $top+$lineheight]
		set scale 100
		set total [expr $inlist+$diff]
		while {$total > $scale} {
		    set scale [expr $scale*10]
		}
		set r1 [expr $gleft+$gwidth*$inlist/$scale]
		set r2 [expr $gleft+$gwidth*$total/$scale]
		.msw.canvas coords ga$line $gleft $top $r1 $bot
		.msw.canvas coords gb$line $r1 $top $r2 $bot
		if {$scale > 100} {
		    set s [expr $scale/100]
		    .msw.canvas coords gt$line [expr $r2+2] [expr $top+1]
		    .msw.canvas itemconfigure gt$line -text "*$s"
		} else {
		    .msw.canvas itemconfigure gt$line -text ""
		}
	    } else {
		set lp [expr -$i]
		.msw.canvas coords ga$lp 0 0 0 0
		.msw.canvas coords gb$lp 0 0 0 0
		.msw.canvas itemconfigure t$lp -text ""
		.msw.canvas itemconfigure gt$lp -text ""
	    }
	}
    } else {
	puts "skipping"
    }
    after 500 updateMemStats $lineheight $gleft $gwidth
}
