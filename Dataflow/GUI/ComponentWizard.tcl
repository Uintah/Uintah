
package require Iwidgets 3.0

proc ComponentWizard { {window .componentWizard} } {
    set w $window 
    if {[winfo exists $w]} {
	destroy $w
    }

    set MAIN_WIDTH 7
    set MAIN_HEIGHT 5.5
    set WIDTH [expr $MAIN_WIDTH - .2]
    set HEIGHT [expr $MAIN_HEIGHT - .7]
    set PAD .1
    set PADi [concat $PAD i]

    global ui_font
    global sci_root
    global modname_font
    global time_font
    
    toplevel $w -width [concat $MAIN_WIDTH i] -height [concat $MAIN_HEIGHT i]
    wm title $w "Component Wizard"

    iwidgets::tabnotebook $w.tabs -width [concat $WIDTH i]\
	                  -height [concat $HEIGHT i] -backdrop [$w cget -background]
    #place $w.tabs -x [concat $PAD i] -y [concat $PAD i] 
    pack $w.tabs -padx $PADi -pady $PADi -fill both -expand yes -side top

    frame $w.buttons
    pack $w.buttons -ipadx $PADi -ipady $PADi -fill both -expand yes -side top

    button $w.buttons.open -text "Open" 
    pack $w.buttons.open -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    button $w.buttons.save -text "Save" 
    pack $w.buttons.save -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    button $w.buttons.create -text "Create" 
    pack $w.buttons.create -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    button $w.buttons.cancel -text "Cancel" -command "destroy $w"  
    pack $w.buttons.cancel -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    set tab1 [$w.tabs add -label "I/O and GUI"]
    canvas $tab1.c -relief sunken -borderwidth 3 -background #038
    place $tab1.c -x .25i -y .25i -width 5i -height 3.5i

    global $tab1.hasgui_value 0
    checkbutton $tab1.hasgui -text "Has GUI" -variable $tab1.hasgui_value\
        -command "eval gui $tab1.c \[set $tab1.hasgui_value\]"
    place $tab1.hasgui -x .25i -y 3.95i -width 1i -height .33i
    checkbutton $tab1.dynamicport -text "Last port is dynamic"
    place $tab1.dynamicport -x 1.5i -y 3.95i -width 2i -height .33i

    make_icon $tab1.c 2.5i 1.75i [set $tab1.hasgui_value]

    set tab2 [$w.tabs add -label "Overview"]
    frame $tab2.f
    combo_listbox $tab2.f.clb "<author name>"
    pack $tab2.f -side top -fill both -expand true
    pack $tab2.f.clb -side top -fill both -expand true

    set tab3 [$w.tabs add -label "Implementation"]
    frame $tab3.f
    
    set testing [$w.tabs add -label "Testing"]
    frame $testing.f
    pack $testing.f -side top -fill both -expand true
    
    prompted_text $testing.f.testing_text "<Insert testing information>" -wrap word -yscrollcommand "$testing.f.sy set"
    pack $testing.f.testing_text -side left -fill both -expand true
    scrollbar $testing.f.sy -orient vert -command "$testing.f.testing_text yview"
    pack $testing.f.sy -side right -fill y

    $w.tabs view "I/O and GUI"
}

proc description_pane {p} {
    frame $p.f
    
}

proc gui {canvas has} {
    set modframe $canvas.moduleFakeModule
    global $modframe.ff.ui
    set p $modframe.ff
    if $has {
        make_ui_button $p
        pack $p.ui -side left -ipadx 5 -ipady 2 -before $p.title
    } else {
        destroy $p.ui
        pack forget $p.ui
    }
}

proc make_ui_button {p} {
    global ui_font
    button $p.ui -text "UI" -borderwidth 2 \
       -anchor center \
          -font $ui_font
}

proc make_icon {canvas modx mody {gui 0} } {
    
    lappend canvases $canvas
    set modframe $canvas.moduleFakeModule
    frame $modframe -relief raised -borderwidth 3
    
    frame $modframe.ff
    pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
    
    set p $modframe.ff
    global modname_font
    global time_font
    
    # Make the title
    prompted_entry $p.title "<click to edit title>" -relief flat \
        -justify center -width 16 -font $modname_font 
    
    # Make the time label
    label $p.time -text "00.00" \
         -font $time_font
    
    # Make the progress graph
    frame $p.inset -relief sunken -height 4 -borderwidth 2 \
	    -width .5i
    frame $p.inset.graph -relief raised -width .5i -borderwidth 2 \
	    -background green
    # Don't pack it in yet - the width is zero... 
    pack $p.inset.graph -fill y -expand yes -anchor nw

    # make a UI button if necessary
    if {$gui} {
        make_ui_button $p
        pack $p.ui -side left -ipadx 5 -ipady 2
    }

    # pack the stuff now
    pack $p.title -side top -padx 2 -anchor w 
    pack $p.time -side left -padx 2
    pack $p.inset -side left -fill y -padx 2 -pady 2
    
    # Stick it in the canvas
    
    $canvas create window $modx $mody -window $modframe \
	    -tags FakeModule -anchor center
}
	
