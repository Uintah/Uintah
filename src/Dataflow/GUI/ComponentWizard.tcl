
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

    global $p.c.moduleFakeModule.iports
    global $p.c.moduleFakeModule.oports
    set $p.c.moduleFakeModule.iports [list]
    set $p.c.moduleFakeModule.oports [list]
    make_icon $p.c 2.5i 1.75i [set $p.hasgui_value]

    frame $p.cmds
    
    set modframe $p.c.moduleFakeModule
    global "$modframe.i ports"
    global "$modframe.o ports"
    set "$modframe.i ports" [list]
    set "$modframe.o ports" [list]
    button $p.cmds.add_iport -text "Add Input Port" -command "eval add_port $modframe i"
    button $p.cmds.add_oport -text "Add Output Port" -command "eval add_port $modframe o"
    pack $p.cmds.add_iport -padx $PADi -pady $PADi \
        -ipadx $PADi -ipady $PADi -expand yes -side top -anchor nw -fill x
    pack $p.cmds.add_oport -padx $PADi -pady $PADi \
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

proc add_port {modframe type} {
    set ports ${type}ports
    global $modframe.$ports
    global portid; 
    if [expr ! [info exists portid]] {
        set portid 0
    }
    incr portid;
    lappend $modframe.$ports componentWizardPort$portid; 
    configPorts $modframe $type
}

proc configPorts {icon type} {
    set ports ${type}ports
    set i 0
    global $icon.$ports
    foreach t [set $icon.$ports] {
        placePort $icon $t $i $type
        incr i
    }
}

proc placePort {icon portnum pos type} {
    set port_width 13
    set port_spacing 18
    set port_height 7
    set portcolor red
    set x [expr $pos * $port_spacing + 6]
    set ports ${type}ports
    set e top
    set port ${type}port${portnum}
    set portlight ${port}light
    if [ expr [lsearch [place slaves $icon] $icon.$port] == -1 ] {
        bevel $icon.$port -width $port_width \
            -height $port_height -borderwidth 3 \
            -edge $e -background $portcolor \
            -pto 2 -pwidth 7 -pborder 2       
        frame $icon.$portlight -width $port_width -height 4 \
        -relief raised -background black -borderwidth 0 
        set menu $icon.$port.menu
        global $menu
        menu $menu -tearoff 0
        $menu add command -label "Edit" -command "edit_port $portnum"
        $menu add command -label "Delete" \
            -command "remove_port $icon $portnum $type"
        bind $icon.$port <ButtonPress-3> "tk_popup $menu %X %Y"
    } else {
        # we may to move the ports around
        place forget $icon.$port
        place forget $icon.$portlight
    }
    if [expr [string compare $type i] == 0] {
        place $icon.$portlight -in $icon.$port \
            -x 0 -rely 1.0 -anchor nw
        place $icon.$port -bordermode outside -x $x -y 0 -anchor nw
    } else {
        place $icon.$portlight -in $icon.$port -x 0 -y 0 -anchor sw
        place $icon.$port -bordermode ignore -rely 1 -anchor sw -x $x
    }
}

proc edit_port {portnum} {
    set w .edit_$portnum
    global $w
    if {[winfo exists $w]} {
	    destroy $w
    }

    toplevel $w

    set f $w.f
    global $f
    frame $f
    
    set lname $w.f.lname
    global $lname
    label $lname -text "Name:"
    grid $lname -column 0 -row 0 -sticky e -padx .1c -pady .1c

    set ename $w.f.ename
    global $ename
    prompted_entry $ename "<port name>"
    grid $ename -column 1 -row 0 -sticky w -padx .1c -pady .1c

    set ldatatype $w.f.ldatatype
    global $ldatatype
    label $ldatatype -text "Datatype:"
    grid $ldatatype -column 0 -row 1 -sticky e -padx .1c -pady .1c
    
    set edatatype $w.f.edatatype
    global $edatatype
    prompted_entry $edatatype "<datatype>"
    grid $edatatype -column 1 -row 1 -sticky w -padx .1c -pady .1c

    set fdescript $w.f.fdescript
    global $fdescript
    frame $fdescript

    set ldescript $fdescript.l
    global $ldescript
    label $ldescript -text "Description:"
    pack $ldescript -side top -anchor w -pady .1c

    set edescript $fdescript.e
    global $edescript
    set sydescript $fdescript.sy
    global $sydescript
    prompted_text $edescript "<description information in HTML>" \
        -wrap word -yscrollcommand "$sydescript set"
    pack $edescript -side left -fill both -expand true
    scrollbar $sydescript -orient vert -command "$edescript yview"
    pack $sydescript -side right -fill y

    grid $fdescript -column 0 -row 3 -columnspan 2 -rowspan 2 -padx .1c -pady .1c

    pack $f -fill both -expand yes
}

proc remove_port {icon portnum type} {
    set port ${type}port${portnum}
    set ports ${type}ports
    global $icon.$ports
    set item_num [lsearch [set $icon.$ports] $portnum]
    place forget $icon.$port
    destroy $icon.${port}light
    destroy $icon.$port
    if [expr $item_num != -1] {
        set $icon.$ports [concat [lrange [set $icon.$ports] 0 [expr $item_num - 1]] \
            [lrange [set $icon.$ports] [expr $item_num + 1] [llength [set $icon.$ports]]]]
        configPorts $icon $type
    }
}

