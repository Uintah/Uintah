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


#package require Iwidgets 3.0

proc ComponentWizard { {window .componentWizard} } {
    set w $window 
    set d $window.data
    set tmpd $window.tmpdata
    global $d
    global $tmpd

    if [info exists $tmpd] {
        unset $tmpd
    }

    array set $tmpd [array get $d]

    if {[winfo exists $w]} {
	    destroy $w
    }

    set MAIN_WIDTH 5
    set MAIN_HEIGHT 6.25
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

    button $w.buttons.save -text "Save" -command "array set $d \[array get $tmpd\]"
    pack $w.buttons.save -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    button $w.buttons.create -text "Create"  \
        -command "array set $d \[array get $tmpd\]"
    pack $w.buttons.create -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    button $w.buttons.cancel -text "Cancel" -command "destroy $w"  
    pack $w.buttons.cancel -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    set io_gui [$w.tabs add -label "I/O and GUI"]
    make_io_gui_pane $io_gui $tmpd

    set overview [$w.tabs add -label "Overview"]
    make_overview_pane $overview $tmpd

    set implementation [$w.tabs add -label "Implementation"]
    make_implementation_pane $implementation $tmpd
    
    set testing [$w.tabs add -label "Testing"]
    make_testing_pane $testing $tmpd
    
    $w.tabs view "I/O and GUI"
}

proc make_io_gui_pane {p d} {
    set PAD .05
    set PADi [concat $PAD i]
    global $d

    canvas $p.c -relief sunken -borderwidth 3 -background #038
    place $p.c -x .125i -y .125i -width 3.25i -height 2.75i

    if [info exists ${d}(hasgui)] {
    } else {
        set ${d}(hasgui) 0
    }
    if [info exists ${d}(dynamicport)] {
    } else {
        set ${d}(dynamicport) 0
    }
    checkbutton $p.hasgui -text "Has GUI" -variable ${d}(hasgui) \
        -command "eval gui $p.c \[set ${d}(hasgui)\]"
    place $p.hasgui -x .25i -y 3i -width 1i -height .33i
    global ${d}.dynamicport
    checkbutton $p.dynamicport -text "Last port is dynamic" -variable ${d}(dynamicport)
    place $p.dynamicport -x 1.3i -y 3i -width 2i -height .33i

    make_icon $p.c 1.5i 1.375i [set ${d}(hasgui)] $d

    frame $p.cmds
    
    set modframe $p.c.moduleFakeModule
    button $p.cmds.add_iport -text "Add Input Port" \
        -command "eval add_port $modframe i $d"
    button $p.cmds.add_oport -text "Add Output Port" \
        -command "eval add_port $modframe o $d"
    button $p.cmds.add_ifile -text "Add Input File" \
        -command ""
    button $p.cmds.add_ofile -text "Add Output File" \
        -command ""
    button $p.cmds.add_idev -text "Add Input Device" \
        -command ""
    button $p.cmds.add_odev -text "Add Output Device" \
        -command ""
    
    if [info exists ${d}(iports)] {
    } else {
        set ${d}(iports) [list]
    }
    if [info exists ${d}(oports)] { 
    } else {
        set ${d}(oports) [list]
    }
    configPorts $modframe "i" $d
    configPorts $modframe "o" $d

    set guidescript $p.guidescript
    create_text_entry $guidescript "Description:" $d guidescript

    set uiinfo $p.uiinfo
    create_text_entry $uiinfo "GUI Info:" $d uiinfo

    pack $p.cmds.add_iport -padx $PADi -pady $PADi \
        -ipady $PADi -expand yes -side top -anchor nw -fill x
    pack $p.cmds.add_oport -padx $PADi -pady $PADi \
        -ipady $PADi -expand yes -side top -anchor nw -fill x
    pack $p.cmds.add_ifile -padx $PADi -pady $PADi \
        -ipady $PADi -expand yes -side top -anchor nw -fill x
    pack $p.cmds.add_ofile -padx $PADi -pady $PADi \
        -ipady $PADi -expand yes -side top -anchor nw -fill x
    pack $p.cmds.add_idev -padx $PADi -pady $PADi \
        -ipady $PADi -expand yes -side top -anchor nw -fill x
    pack $p.cmds.add_odev -padx $PADi -pady $PADi \
        -ipady $PADi -expand yes -side top -anchor nw -fill x

    pack $guidescript -fill x -side bottom -anchor sw \
        -padx $PADi 
    pack $uiinfo -fill x -side bottom -anchor sw \
        -padx $PADi 
    pack $p.cmds -expand no -side right -anchor ne -padx $PADi -pady $PADi
}

proc make_overview_pane {p d} {
    global $d

    set authors $p.authors
    create_clb_entry $authors "Authors:" "<enter new author here>" $d authors
    set summary $p.summary
    create_text_entry $summary "Summary:" $d summary

    set descript $p.descript
    create_text_entry $descript "Description:" $d descript

    set lexamplesr $p.lexamplesr
    label $lexamplesr -text "Example Source: "

    set eexamplesr $p.eexamplesr
    prompted_entry $eexamplesr "<enter example source filename>" "
            global $d;
            set ${d}(examplesr) \[get_prompted_entry $eexamplesr\];
        " 
    if [info exists ${d}(examplesr)] {
        set_prompted_entry $p.eexamplesr [set ${d}(examplesr)]
    }

    pack $authors -side top -fill both -expand true -padx .1c -pady .1c
    pack $summary -side top -anchor w -fill x -expand true -padx .1c -pady .1c
    pack $descript -side top -anchor w -fill x -expand true -padx .1c -pady .1c
    pack $lexamplesr -side left -anchor e -padx .1c -pady .1c
    pack $eexamplesr -side left -anchor w -padx .1c -pady .1c

}

proc make_implementation_pane {p d} {
    global $d

    set ccfiles $p.ccfiles
    create_clb_entry $ccfiles ".cc files:" "<enter the basename of a .cc file>" \
        $d ccfiles

    set cfiles $p.cfiles
    create_clb_entry $cfiles ".c files:" "<enter the basename of a .c file>" \
        $d cfiles
        
    set ffiles $p.ffiles
    create_clb_entry $ffiles ".f files:" "<enter the basename of a .f file>" \
        $d ffiles

    pack $ccfiles -side top -fill both -expand true -padx .1c -pady .1c
    pack $cfiles -side top -fill both -expand true -padx .1c -pady .1c
    pack $ffiles -side top -fill both -expand true -padx .1c -pady .1c
}

proc make_testing_pane {p d} {
    global $d

    set testing $p.testing
    create_text_entry $testing "Testing Info:" $d testing

    pack $testing -side top -expand yes -fill both 
}

proc create_clb_entry {f label prompt array index} {
    frame $f -relief ridge -borderwidth .1c

    set l $f.l
    label $l -text $label
    pack $l -side top -anchor w -pady .1c 
    global $array

    set clb $f.clb
    combo_listbox $clb $prompt \
        "global $array;
        set ${array}($index) \[$clb.listbox get 0 end\]"
    if [info exists ${array}($index)] {
        foreach entry [set ${array}($index)] {
            $clb.listbox insert end $entry
        }
    }
    pack $clb -side top -anchor n -fill x -expand yes
}

proc create_text_entry {f label array index} {
    frame $f
    set l $f.l
    set t $f.t
    set sy $f.sy
    global $array

    label $l -text $label
    text $t -wrap word -yscrollcommand "$sy set" -height 5
    if [info exists ${array}($index)] {
       $t insert 1.0 [set ${array}($index)]  
    }
    bindtags $t [concat [bindtags $t] GrabText$t]
    bind GrabText$t <Key> "
        global $array;
        set ${array}($index) \[%W get 1.0 end\]
    "
    scrollbar $sy -orient vert -command "$t yview"

    pack $l -side top -anchor w
    pack $t -side left -fill both -expand true
    pack $sy -side right -fill y
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

proc make_icon {canvas modx mody {gui 0} d} {
    global $d
    
    #lappend canvases $canvas
    set modframe $canvas.moduleFakeModule
    frame $modframe -relief raised -borderwidth 3
    
    frame $modframe.ff
    pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
    
    set p $modframe.ff
    global modname_font
    global time_font
    
    # Make the title
    if [info exists ${d}(title)] {
    } else {
        set ${d}(title) ""
    }
    prompted_entry $p.title "<click to edit title>" "
            global $p.title.real_text;
            set ${d}(title) \[set $p.title.real_text\];
        " -relief flat -justify center -width 16 -font $modname_font 
    set_prompted_entry $p.title [set ${d}(title)]

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

proc add_port {modframe type d} {
    set ports ${type}ports
    global portid; 
    global $d
    if [expr ! [info exists portid]] {
        set portid 0
    }
    incr portid;
    set portnum componentWizardPort$portid
    global $portnum
    lappend ${d}($ports) $portnum
    configPorts $modframe $type $d
}

proc configPorts {icon type d} {
    set ports ${type}ports
    set i 0
    global $d
    foreach t [set ${d}($ports)] {
        placePort $icon $t $i $type $d
        incr i
    }
}

proc placePort {icon portnum pos type d} {
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
            -command "remove_port $icon $portnum $type $d"
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
    global $portnum
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
    if [info exists ${portnum}(name)] {
        global $ename.real_text
        set_prompted_entry $ename [set ${portnum}(name)]
    }
    grid $ename -column 1 -row 0 -sticky w -padx .1c -pady .1c

    set ldatatype $w.f.ldatatype
    global $ldatatype
    label $ldatatype -text "Datatype:"
    grid $ldatatype -column 0 -row 1 -sticky e -padx .1c -pady .1c
    
    set edatatype $w.f.edatatype
    global $edatatype
    prompted_entry $edatatype "<datatype>"
    if [info exists ${portnum}(datatype)] {
        set_prompted_entry $edatatype [set ${portnum}(datatype)]
    }
    grid $edatatype -column 1 -row 1 -sticky w -padx .1c -pady .1c

    set fdescript $w.fdescript
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
    prompted_text $edescript "<description information in HTML>" "" \
        -wrap word -yscrollcommand "$sydescript set"
    pack $edescript -side left -fill both -expand true
    if [info exists ${portnum}(descript)] {
        set_prompted_text $edescript [set ${portnum}(descript)]
    }
    scrollbar $sydescript -orient vert -command "$edescript yview"
    pack $sydescript -side right -fill y

    set fcompnames $w.fcompnames
    global $fcompnames
    frame $fcompnames -relief ridge -borderwidth .1c

    set lcompnames $fcompnames.lcompnames
    global $lcompnames
    label $lcompnames -text "Component Names:"
    pack $lcompnames -side top -anchor w -pady .1c 

    set clbcompnames $fcompnames.clbcompnames
    global $clbcompnames
    combo_listbox $clbcompnames "<enter new component names here>"
    if [info exists ${portnum}(compnames)] {
        global $clbcompnames.listbox
        foreach compname [set ${portnum}(compnames)] {
            $clbcompnames.listbox insert end $compname
        }
    }
    pack $clbcompnames -side left -fill both -expand yes

    set fbuttons $w.fcubbonts
    global $fbuttons
    frame $fbuttons 

    set save $fbuttons.save
    global $save
    button $save -text OK -command "save_port_edit $portnum ; destroy $w"
    pack $save -side left -padx .1c -pady .1c -ipadx .1c -ipady .1c
        

    set cancel $fbuttons.cancel
    global $cancel
    button $cancel -text Cancel -command "destroy $w"
    pack $cancel -side left -padx .1c -pady .1c -ipadx .1c -ipady .1c

    pack $f -fill both -expand yes -side top
    pack $fdescript -fill both -expand yes -side top
    pack $fcompnames -fill both -expand yes -side top
    pack $fbuttons -fill both -expand yes -side top

    focus $w
    grab $w
    tkwait variable $w
}

proc save_port_edit {portnum} {
    global $portnum
    set ${portnum}(name) [get_prompted_entry .edit_$portnum.f.ename] 
    set ${portnum}(datatype) [get_prompted_entry .edit_$portnum.f.edatatype] 
    set ${portnum}(descript) [get_prompted_text .edit_$portnum.fdescript.e] 
    set ${portnum}(compnames) [.edit_$portnum.fcompnames.clbcompnames.listbox get 0 end] 
}

proc remove_port {icon portnum type d} {
    set port ${type}port${portnum}
    set ports ${type}ports
    global $d
    set item_num [lsearch [set ${d}($ports)] $portnum]
    place forget $icon.$port
    destroy $icon.${port}light
    destroy $icon.$port
    if [expr $item_num != -1] {
        set ${d}($ports) [concat [lrange [set ${d}($ports)] 0 [expr $item_num - 1]] \
            [lrange [set ${d}($ports)] [expr $item_num + 1] \
                    [llength [set ${d}($ports)]]]]
        configPorts $icon $type $d
    }
}

