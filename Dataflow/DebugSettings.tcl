
proc showDebugSettings {} {
    global debugsettings_window
    if [catch {raise $debugsettings_window}] {
	toplevel .dsw
	wm title .dsw "Debug Settings"
	wm iconname .dsw "DebugSettings"
	wm minsize .dsw 100 100
	set DebugSettings_window .dsw
	canvas .dsw.canvas -yscroll ".dsw.vscroll set" \
		-scrollregion {0c 0c 8c 50c} \
		-width 8c -height 8c
	scrollbar .dsw.vscroll -relief sunken -command ".dsw.canvas yview" \
		-foreground plum2 -activeforeground SteelBlue2
	pack .dsw.vscroll -side right -fill y -padx 4 -pady 4
	pack .dsw.canvas -expand yes -fill y -pady 4
	set lineheight [winfo pixels .dsw.canvas 8p]
	set tleft [winfo pixels .dsw.canvas 1.1c]
	set gleft [winfo pixels .dsw.canvas 5.5c]
	set gwidth [winfo pixels .dsw.canvas 2c]
	set font -Adobe-courier-medium-r-*-80-75-*
    }
    
    #  proc updateDebugSettings {lineheight tleft gleft gwidth font old_ntasks}
    if {[winfo exists .dsw] == 0} {
	return
    }
    set ds [debugsettings]
    puts $ds
}
proc junk {} {
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
