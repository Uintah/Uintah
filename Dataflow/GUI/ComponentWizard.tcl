
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
    wm resizable $w 0 0

    iwidgets::tabnotebook $w.tabs -width [concat $WIDTH i]\
	                  -height [concat $HEIGHT i] -backdrop [$w cget -background]
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

    set io_gui [$w.tabs add -label "I/O and GUI"]
    make_io_gui_pane $io_gui

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

proc make_io_gui_pane {p} {
    set PAD .1
    set PADi [concat $PAD i]

    canvas $p.c -relief sunken -borderwidth 3 -background #038
    place $p.c -x .25i -y .25i -width 5i -height 3.5i

    global $p.hasgui_value 0
    checkbutton $p.hasgui -text "Has GUI" -variable $p.hasgui_value\
        -command "eval gui $p.c \[set $p.hasgui_value\]"
    place $p.hasgui -x .25i -y 3.95i -width 1i -height .33i
    checkbutton $p.dynamicport -text "Last port is dynamic"
    place $p.dynamicport -x 1.5i -y 3.95i -width 2i -height .33i

    make_icon $p.c 2.5i 1.75i [set $p.hasgui_value]

    frame $p.cmds
    
    set modframe $p.c.moduleFakeModule
    global $modframe.iports
    set $modframe.iports [list]
    button $p.cmds.add_port -text "Add Input Port" \
        -command "global $modframe.iports; \
                  set port $modframe.iport_info\[llength \[set $modframe.iports\]\]; \
                  global \$port; \
                  lappend $modframe.iports \$port; \
                  configIPorts $modframe"
    pack $p.cmds.add_port -padx $PADi -pady $PADi \
        -ipadx $PADi -ipady $PADi -expand yes -side top -anchor nw -fill x

    pack $p.cmds -expand no -side right -anchor ne -padx $PADi -pady $PADi
}

proc description_pane {p} {
    frame $p
    
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
    
    #lappend canvases $canvas
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

proc configIPorts {icon} {
    set port_width 13
    set port_spacing 18
    set port_height 7
    set i 0
    global $icon.iports
    foreach t [set $icon.iports] {
        puts $t
        set portcolor red
        set x [expr $i * $port_spacing + 6]
        set e top
        if [ expr [lsearch [place slaves $icon] $icon.iport$i] == -1 ] {
            bevel $icon.iport$i -width $port_width \
                -height $port_height -borderwidth 3 \
                -edge $e -background $portcolor \
                -pto 2 -pwidth 7 -pborder 2       
            place $icon.iport$i -bordermode outside -x $x -y 0 -anchor nw
        }
        incr i
    }
}
	
