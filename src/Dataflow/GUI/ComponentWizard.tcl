
package require Iwidgets 3.0

proc ComponentWizard {} {
    set w .componentWizard
    if {[winfo exists $w]} {
	destroy $w
    }

    set MAIN_WIDTH 7
    set MAIN_HEIGHT 5.5
    set WIDTH [expr $MAIN_WIDTH - .2]
    set HEIGHT [expr $MAIN_HEIGHT - .7]
    set PAD .1

    global ui_font
    global sci_root
    global modname_font
    global time_font
    
    toplevel $w -width [concat $MAIN_WIDTH i] -height [concat $MAIN_HEIGHT i]
    wm title $w "Component Wizard"

    iwidgets::tabnotebook $w.tabs -width [concat $WIDTH i]\
	                  -height [concat $HEIGHT i]
    place $w.tabs -x [concat $PAD i] -y [concat $PAD i] 

    button $w.open -text "Open" 
    place $w.open -x [concat $PAD i] -y [expr $HEIGHT + [expr 2* $PAD]]i\
          -width 1i -height .33i

    button $w.save -text "Save" 
    place $w.save -x [concat [expr $PAD + 1] i]\
	  -y [expr $HEIGHT + [expr 2* $PAD]]i\
          -width 1i -height .33i

    button $w.create -text "Create" 
    place $w.create -x [concat [expr $PAD + 2.25] i]\
	  -y [expr $HEIGHT + [expr 2* $PAD]]i\
          -width 1i -height .33i

    button $w.cancel -text "Cancel" 
    place $w.cancel -x [concat [expr $PAD + 3.5] i]\
	  -y [expr $HEIGHT + [expr 2* $PAD]]i\
          -width 1i -height .33i

    set tab1 [$w.tabs add -label "I/O and GUI"]
    canvas $tab1.c -relief sunken -borderwidth 3 -background #038
    place $tab1.c -x .25i -y .25i -width 5i -height 3.5i
    make_icon $tab1.c 2.5i 1.75i

    checkbutton $tab1.hasgui -text "Has GUI"
    place $tab1.hasgui -x .25i -y 3.95i -width 1i -height .33i
    checkbutton $tab1.dynamicport -text "Last port is dynamic"
    place $tab1.dynamicport -x 1.5i -y 3.95i -width 2i -height .33i

    set tab2 [$w.tabs add -label "Overview"]
    frame $tab2.f

    set tab3 [$w.tabs add -label "Implementation"]
    frame $tab3.f
    
    set tab4 [$w.tabs add -label "Testing"]
    frame $tab4.f

    $w.tabs view "I/O and GUI"
}

proc make_icon {canvas modx mody} {
    
    lappend canvases $canvas
    set modframe $canvas.moduleFakeModule
    frame $modframe -relief raised -borderwidth 3
    
    frame $modframe.ff
    pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
    
    set p $modframe.ff
    global ui_font
    global sci_root
    button $p.ui -text "UI" -borderwidth 2 \
	   -anchor center \
          -font $ui_font
    pack $p.ui -side left -ipadx 5 -ipady 2

    global modname_font
    global time_font
    
    #
    # Make the title
    #
    entry $p.title -relief flat -justify center -width 16 \
         -font $modname_font 
    pack $p.title -side top -padx 2 -anchor w 
    
    #
    # Make the time label
    #
    label $p.time -text "00.00" \
         -font $time_font
    pack $p.time -side left -padx 2
    
    #
    # Make the progress graph
    #
    frame $p.inset -relief sunken -height 4 -borderwidth 2 \
	    -width .5i
    pack $p.inset -side left -fill y -padx 2 -pady 2
    frame $p.inset.graph -relief raised -width .5i -borderwidth 2 \
	    -background green
    # Don't pack it in yet - the width is zero... 
    pack $p.inset.graph -fill y -expand yes -anchor nw

    #
    # Stick it in the canvas
    #
    
    $canvas create window $modx $mody -window $modframe \
	    -tags FakeModule -anchor center
}
	
