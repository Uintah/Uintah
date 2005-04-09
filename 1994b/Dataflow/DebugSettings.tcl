
proc showDebugSettings {} {
    global debugsettings_window
    if [catch {raise $debugsettings_window}] {
	toplevel .dsw
	wm title .dsw "Debug Settings"
	wm iconname .dsw "DebugSettings"
	wm minsize .dsw 100 100
	set DebugSettings_window .dsw

	canvas .dsw.canvas -yscroll ".dsw.scroll set" -borderwidth 0 \
		-scrollregion {0c 0c 8c 50c} \
		-width 8c -height 8c
	pack .dsw.canvas -side left -padx 2 -pady 2 -fill both -expand yes

	frame .dsw.canvas.frame -borderwidth 2
	frame .dsw.canvas.frame.l
	pack .dsw.canvas.frame.l -side left -anchor w -padx 4
	frame .dsw.canvas.frame.r
	pack .dsw.canvas.frame.r -side right -anchor w
	pack .dsw.canvas.frame
	.dsw.canvas create window 0 0 -window .dsw.canvas.frame -anchor nw

	scrollbar .dsw.scroll -relief sunken -command ".dsw.canvas yview"
	pack .dsw.scroll -fill y -side right -padx 2 -pady 2

	set ds [debugsettings]
	
	for {set i 0} {$i < [llength $ds]} {incr i 1} {
	    set dbg [lindex $ds $i]
	    set module [lindex $dbg 0]
	    set items [lindex $dbg 1]
	    frame .dsw.canvas.frame.l.$i
	    label .dsw.canvas.frame.l.$i.module -text $module
	    pack .dsw.canvas.frame.l.$i.module -side left -padx 2 -pady 2 -anchor w
	    frame .dsw.canvas.frame.r.$i
	    for {set j 0} {$j < [llength $items]} {incr j 1} {
		set item [lindex $items $j]
		checkbutton .dsw.canvas.frame.r.$i.$j -text $item -variable $module,$item \
			-relief groove
		puts "varname is $dbg,$item"
		pack .dsw.canvas.frame.r.$i.$j -side left -padx 2 -pady 2 -anchor w
	    }
	    pack .dsw.canvas.frame.l.$i -anchor w
	    pack .dsw.canvas.frame.r.$i -anchor w
	}
    }
    
    #  proc updateDebugSettings {lineheight tleft gleft gwidth font old_ntasks}
    if {[winfo exists .dsw] == 0} {
	return
    }
}
proc junk {} {
    canvas .dsw.canvas -yscroll ".dsw.vscroll set" \
	    -scrollregion {0c 0c 8c 50c} \
	    -width 8c -height 8c
    scrollbar .dsw.vscroll -relief sunken -command ".dsw.canvas yview"
    pack .dsw.vscroll -side right -fill y -padx 4 -pady 4
    pack .dsw.canvas -expand yes -fill y -pady 4
    set lineheight [winfo pixels .dsw.canvas 8p]
    set tleft [winfo pixels .dsw.canvas 1.1c]
    set gleft [winfo pixels .dsw.canvas 5.5c]
    set gwidth [winfo pixels .dsw.canvas 2c]
    set font -Adobe-courier-medium-r-*-80-75-*

    foreach i [split [debugsettings]] {
	set info [split [debugsettings debug $i] "|"]
	set module [lindex $info 0]
	set nvars [lindex $info 1]
	set top [expr $i*$lineheight*3]
	set bot [expr $top+$lineheight*2]
	set g1 [expr $gleft+$gwidth*$module/$nvars]
	set gright [expr $gleft+$gwidth]
	.dsw.canvas coords ga$i $gleft $top $g1 $bot
	.dsw.canvas coords gb$i $g1 $top $gright $bot
	.dsw.canvas itemconfigure ta$i -text "$module$k"
	.dsw.canvas itemconfigure tb$i -text "$nvars$k"
	.dsw.canvas itemconfigure t$i -text $title
    }
}
