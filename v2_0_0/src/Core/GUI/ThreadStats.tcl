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


proc showThreadStats {} {
    global threadstats_window
    if [catch {raise $threadstats_window}] {
	toplevel .tsw
	wm title .tsw "Thread Statistics"
	wm iconname .tsw "ThreadStats"
	wm minsize .tsw 100 100
	set threadstats_window .tsw
	canvas .tsw.canvas -yscrollcommand ".tsw.vscroll set" \
		-scrollregion {0c 0c 8c 50c} \
		-width 8c -height 8c
	scrollbar .tsw.vscroll -relief sunken -command ".tsw.canvas yview"
	pack .tsw.vscroll -side right -fill y -padx 4 -pady 4
	pack .tsw.canvas -expand yes -fill y -pady 4
	set lineheight [winfo pixels .tsw.canvas 8p]
	set tleft [winfo pixels .tsw.canvas 1.2c]
	set gleft [winfo pixels .tsw.canvas 5.5c]
	set gwidth [winfo pixels .tsw.canvas 1.9c]
	set font -Adobe-courier-bold-r-*-100-75-*
	set ntasks 0
    }
    after 0 updateThreadStats $lineheight $tleft $gleft $gwidth $font $ntasks
}

proc updateThreadStats {lineheight tleft gleft gwidth font old_ntasks} {
    if {[winfo exists .tsw] == 0} {
	return
    }
    set ntasks [threadstats ntasks]
    if {$ntasks < $old_ntasks} {
	# Delete a few...
	for {set i $ntasks} {$i<$old_ntasks} {incr i} {
	    .tsw.canvas delete t$i b$i ga$i gb $i ta$i tb$i
	    .tsw.b$i delete
	}
    } else {
	# add some more...
	for {set i $old_ntasks} {$i<$ntasks} {incr i} {
	    set top [expr $i*$lineheight*3+4]
	    set mid [expr $top+$lineheight/2]
	    button .tsw.b$i -text DBX -command "threadstats dbx $i" \
		    -font $font -borderwidth 3 -padx 4 -pady 2
	    .tsw.canvas create window 4 $top -window .tsw.b$i -tags b$i \
		    -anchor nw
	    .tsw.canvas create text $tleft $mid -tag t$i -font $font -anchor nw
	    .tsw.canvas create text [expr $gleft+$gwidth+2] $top \
		    -tag ta$i -font $font -anchor nw
	    .tsw.canvas create text [expr $gleft+$gwidth+2] [expr $top+$lineheight] \
		    -tag tb$i -font $font -anchor nw
	    .tsw.canvas create rectangle 0 0 0 0 -tag ga$i -fill red
	    .tsw.canvas create rectangle 0 0 0 0 -tag gb$i -fill blue
	}
    }
    set k k
    foreach i [split [threadstats changed]] {
	set info [split [threadstats thread $i] "|"]
	set stack [lindex $info 0]
	set tstack [lindex $info 1]
	set title [lindex $info 2]
	set top [expr $i*$lineheight*3+4]
	set bot [expr $top+$lineheight*2]
	set g1 [expr $gleft+$gwidth*$stack/$tstack]
	set gright [expr $gleft+$gwidth]
	.tsw.canvas coords ga$i $gleft $top $g1 $bot
	.tsw.canvas coords gb$i $g1 $top $gright $bot
	.tsw.canvas itemconfigure ta$i -text "$stack$k"
	.tsw.canvas itemconfigure tb$i -text "$tstack$k"
	.tsw.canvas itemconfigure t$i -text $title
    }
    after 1000 updateThreadStats $lineheight $tleft $gleft $gwidth $font $ntasks
}
