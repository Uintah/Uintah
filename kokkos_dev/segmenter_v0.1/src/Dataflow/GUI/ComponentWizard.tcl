#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#



package require Iwidgets 3.0

proc ComponentWizard { {window .componentWizard} } {
    set w $window 
    set d $window.data

    if {[winfo exists $w]} {
        moveToCursor $w    
        SciRaise $w
        return
    }

    set tmpd $window.tmpdata
    global $d
    global $tmpd

    if [info exists $tmpd] {
        unset $tmpd
    }

    array set $tmpd [array get $d]

    set PAD .1
    set PADi [concat $PAD i]

    global ui_font
    global sci_root
    global modname_font
    global time_font
    
    toplevel $w
    wm withdraw $w; # immediately withdraw it to avoid flicker

    wm title $w "Component Wizard"
    wm minsize .componentWizard 320 550

    # Override the 'destroy' button on the window decoration to just close the window.
    wm protocol $w WM_DELETE_WINDOW "wm withdraw $w"

    # Tab panel
    iwidgets::tabnotebook $w.tabs 
    pack $w.tabs -fill both -expand yes

    # Horizontal separator
    frame $w.separator -height 2 -relief sunken -borderwidth 2
    pack  $w.separator -fill y -padx 5 -pady 5 -expand no -fill x

    # Close / Create buttons
    frame $w.buttons
    pack $w.buttons -ipadx $PADi -ipady $PADi -fill x -expand no


    button $w.buttons.close -text "Close" -command "wm withdraw $w"
    button $w.buttons.help -text "Help" -command "showHelp"
    button $w.buttons.create -text "Create" -command "array set $d \[array get $tmpd\]; generateXML $d"

    pack $w.buttons.close $w.buttons.help $w.buttons.create -side left -expand yes -fill x -padx $PADi

    set io_gui [$w.tabs add -label "I/O"]
    make_io_gui_pane $io_gui $tmpd

    set overview [$w.tabs add -label "Description"]
    make_description_pane $overview $tmpd

    $w.tabs view "I/O"

    moveToCursor $w "leave_up"
}

proc showHelp {} {
    createSciDialog \
        -title "Component Wizard Help" \
        -message [join [concat {"Component Wizard\n\nThis wizard allows you to create the skeleton for a new module."} \
                            {"The .cc file will be added to your source tree.  You will then"} \
                            {"have to exit SCIRun, edit the new module file to include your code,"} \
                            {"re-compile, and then re-run SCIRun."} \
                            {""}] \n] \
        -info
}

proc make_io_gui_pane {p d} {
    set PAD .05
    set PADi [concat $PAD i]
    global $d

    # Create the main frames  (Canvas Frame (cf), Command Frame (cmds), Check Button Frame (cbf))
    frame $p.cf
    frame $p.cont
    frame $p.cont.cmds 
    frame $p.cont.cbf  

    # Create the canvas

    canvas $p.cf.c -width 3i -height 3i -relief sunken \
	-borderwidth 3 -background #036
    pack $p.cf.c -fill both -expand yes

    if { ![info exists ${d}(hasgui)] } {
        set ${d}(hasgui) 0
    }
    if { ![info exists ${d}(dynamicport)] } {
        set ${d}(dynamicport) 0
    }

    checkbutton $p.cont.cbf.hasgui -text "Has GUI" -variable ${d}(hasgui) \
        -command "eval gui $p.cf.c \[set ${d}(hasgui)\]"
    TooltipMultiline $p.cont.cbf.hasgui \
        "Adds the 'UI' button to the module and creates a 'method ui {}' to the module's\n" \
        ".tcl file.  NOTE: The module creator must fill in the 'guts' of this method in\n" \
        "order to create the actual GUI."

    global ${d}.dynamicport
    checkbutton $p.cont.cbf.dynamicport -text "Last port is dynamic" -variable ${d}(dynamicport)
    TooltipMultiline $p.cont.cbf.dynamicport \
        "If selected, the last input port will be dynamic.  This means that\n" \
        "any time a pipe is connected to it, another new port will be dynamically\n" \
        "created.  This allows an 'infinite' number of inputs to this module."

    pack $p.cont.cbf.dynamicport $p.cont.cbf.hasgui -side top -anchor w

    make_icon $p.cf.c 1.0i .375i [set ${d}(hasgui)] $d

    set modframe $p.cf.c.moduleFakeModule
    button $p.cont.cmds.add_iport -text "Add Input Port"  -command "eval add_port $modframe i $d"
    button $p.cont.cmds.add_oport -text "Add Output Port" -command "eval add_port $modframe o $d"

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


    frame $p.cp

    frame $p.cp.name
    label $p.cp.name.label -text "Module Name:" -width 1
    entry $p.cp.name.entry -textvar ${d}(title) 

    TooltipMultiWidget "$p.cp.name.label $p.cp.name.entry" \
        "The name for this module.  Names begin with a capital letter and must not contain spaces."

    frame $p.cp.pack
    label $p.cp.pack.label -text "Package:" -width 1
    entry $p.cp.pack.entry -textvar ${d}(package) 

    TooltipMultiWidget "$p.cp.pack.label $p.cp.pack.entry" \
        "The Pacakge in which to place this module.  Examples: SCIRun, BioPSE, MatlabInterface, etc..."

    frame $p.cp.cat
    label $p.cp.cat.label -text "Category:" -width 1
    entry $p.cp.cat.entry -textvar ${d}(category) 

    TooltipMultiWidget "$p.cp.cat.label $p.cp.cat.entry" \
        "The Category in which to place this module.  Examples: DataIO, Math, Visualization, etc..."

    pack $p.cont.cmds.add_iport -padx $PADi -pady $PADi -ipady $PADi -expand no -side top -fill x
    pack $p.cont.cmds.add_oport -padx $PADi -pady $PADi -ipady $PADi -expand no -side top -fill x


    pack $p.cp.pack.label $p.cp.pack.entry -side left -fill x -expand yes -anchor w
    pack $p.cp.cat.label $p.cp.cat.entry -side left -fill x -expand yes -anchor w
    pack $p.cp.name.label $p.cp.name.entry -side left -fill x -expand yes -anchor w

    pack $p.cp.name $p.cp.cat $p.cp.pack -anchor w -side top -fill x -expand yes     
    trace variable ${d}(title)  w "update_title_entry_bind"

    ### Packing:
    #    .cf (Canvas Frame) | .cont.cmds (Command Frame)
    #    .cbf (Check Buttons Frame)
    #    .cp (Name entries)
    #

    pack $p.cont.cmds $p.cont.cbf -side left -fill x -expand yes
    pack $p.cf -fill both -expand yes
    pack $p.cont -side top -fill x -expand yes -anchor e
    pack $p.cp -side top -fill both -expand yes -anchor w


#    grid $p.cf   -sticky news -padx 5 -pady 2
#    grid $p.cmds -column 1 -row 0 -sticky nws -pady 2
#    grid $p.cbf  -sticky ew -padx 5 -pady 2 -ipadx 5 -ipady 2
#    grid $p.cp   -columnspan 2 -ipady 5 -pady 5 -padx 5 -sticky ew

}

################################################################
#
# update_title_entry_bind name1 name2 op
#
#    name1, name2, and op are provided by the 'trace' command (and are not used).
#
proc update_title_entry_bind { name1 name2 op } {
    global .componentWizard.tmpdata

    set p [.componentWizard.tabs childsite "I/O"]
    set title_pentry $p.cf.c.moduleFakeModule.ff.title
    set tmp [set .componentWizard.tmpdata(title)]

    set_prompted_entry $title_pentry $tmp
}


proc make_description_pane {p d} {
    global $d

    set authors $p.authors
    create_clb_entry $authors "Authors:" "<enter new author here>" $d authors

    set summary $p.summary
    create_text_entry $summary "Summary:" $d summary \
        [join [concat {"This information will be stored in the Summary section of the module help."} \
                    {"It should provide a general description of the purpose of the module."}\
                    {"You may manually enter this information later in the module's XML file."}] \n]

    set descript $p.descript
    create_text_entry $descript "Description:" $d descript \
        [join [concat {"This information will be stored in the Description section of the module help."} \
                    {"It should describe in detail what the purpose of the module is and how the"}\
                    {"module performs its job.  You may manually enter this information"}\
                    {"later in the module's XML file."}] \n]


    pack $authors  -side top -fill both -expand true -padx .1c -pady .1c
    pack $summary  -side top -anchor w -fill x -expand true -padx .1c -pady .1c
    pack $descript -side top -anchor w -fill x -expand true -padx .1c -pady .1c
}


proc create_clb_entry {f label prompt array index} {
    frame $f 

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

########################################################################
#
# Sets the state on a text_entry created using the create_text_entry procedure.
# State may be either "normal" or "disable"
#
proc text_entry_set_state { f state } {
    $f.l config -state $state
    $f.t config -state $state
}

proc create_text_entry {f label array index {tip "" }} {

    frame $f
    set l $f.l
    set t $f.t
    set sy $f.sy
    global $array

    label $l -text $label

    if { $tip != "" } {
        Tooltip $l $tip
    }

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
    if { ![info exists ${d}(title)] } {
        set ${d}(title) ""
    }

    prompted_entry $p.title "<click to edit name>" "
            global $p.title.real_text;
            set ${d}(title) \[set $p.title.real_text\];
        " -relief flat -justify left -width 17 -font $modname_font 

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
    
    $canvas create window $modx $mody -window $modframe -tags FakeModule -anchor n
}

proc add_port {modframe type d} {
    set ports ${type}ports
    global portid; 
    global $d
    if {! [info exists portid]} {
        set portid 0
    }
    incr portid;
    set portnum componentWizardPort$portid
    global $portnum
    lappend ${d}($ports) $portnum
    configPorts $modframe $type $d

    edit_port $portnum
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
    if { [string compare $type i] == 0 } {
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
    wm withdraw $w; # immediately withdraw it to avoid flicker

    wm title $w "Edit Port Information"
   ### wm minsize $w 400 200

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

    TooltipMultiWidget "$lname $ename" \
        [join [concat {"The name of the port that will appear when the mouse"} \
                   {"hovers over this port on the module on the network canvas."}] \n]

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

    TooltipMultiWidget "$ldatatype $edatatype" \
        [join [concat {"The type of data that this port will send/receive."} \
                   {"Examples:  SCIRun::Matrix, SCIRun::Geometry, SCIRun::Field..."}] \n]


    set fbuttons $w.fcubbonts
    global $fbuttons
    frame $fbuttons 

    set save $fbuttons.save
    set close $fbuttons.close
    global $close
    global $save

    button $save -text OK -command "save_port_edit $portnum ; destroy $w"
    button $close -text Close -command "destroy $w"

    pack $close $save -side right -padx 5 -pady 2 -ipadx .1c -ipady 2

    pack $f -fill x -expand yes -side top -padx 2 -pady 2
    pack $fbuttons -fill both -expand yes -side top

    moveToCursor $w "leave_up"

    focus $w
    grab $w

    tkwait variable $w
}

proc save_port_edit {portnum} {
    global $portnum
    set ${portnum}(name) [get_prompted_entry .edit_$portnum.f.ename] 
    set ${portnum}(datatype) [get_prompted_entry .edit_$portnum.f.edatatype] 
}

proc remove_port {icon portnum type d} {
    set port ${type}port${portnum}
    set ports ${type}ports
    global $d
    set item_num [lsearch [set ${d}($ports)] $portnum]
    place forget $icon.$port
    destroy $icon.${port}light
    destroy $icon.$port
    if { $item_num != -1 } {
        set ${d}($ports) [concat [lrange [set ${d}($ports)] 0 [expr $item_num - 1]] \
            [lrange [set ${d}($ports)] [expr $item_num + 1] \
                    [llength [set ${d}($ports)]]]]
        configPorts $icon $type $d
    }
}

proc generateXML { d } {
    global $d
    set id [open cwmmtemp.xml {WRONLY CREAT TRUNC}]

    ######################################################################
    # Make sure that the user has entered all the necessary data!
    if { ![info exists ${d}(title)] || ![llength [set ${d}(title)]] } {
      createSciDialog -title "Module Creation Error" -message "Please enter a 'Module Name'" -error
      return
    }
    if { ![info exists ${d}(package)] || ![llength [set ${d}(package)]] } {
      createSciDialog -title "Module Creation Error" -message "Please enter a 'Package'" -error
      return
    }
    if { ![info exists ${d}(category)] || ![llength [set ${d}(category)]] } {
      createSciDialog -title "Module Creation Error" -message "Please enter a 'Category'" -error
      return
    }
    if { ![info exists ${d}(summary)] || ![llength [set ${d}(summary)]] } {
      createSciDialog -title "Module Creation Error" -message "Please enter a 'Summary' (On the 'Description' tab)" -error
      return
    }
    if { ![info exists ${d}(descript)] || ![llength [set ${d}(descript)]] } {
      createSciDialog -title "Module Creation Error" -message "Please enter a 'Description' (On the 'Description' tab)" -error
      return
    }
    #
    ######################################################################

    # Make sure module name does not have spaces and starts with a Capital letter.
    set title [set ${d}(title)]
    if { [string first " " $title] != -1  } {
        createSciDialog -title "Module Creation Error: ($title)" \
            -message [join [concat {"The name of the module has a space in it.  This is not allowed."} \
                                   {"Please fix this and then create the module."}] \n] -error
        return
    }
    set firstLetter [string toupper [string index $title 0]]
    set ${d}(title) $firstLetter[string range $title 1 end]

    puts $id "<?xml version=\"1.0\" encoding=\"UTF-8\" ?>"
    puts $id "<!DOCTYPE component SYSTEM \"../src/Dataflow/XML/component.dtd\">"

    puts $id "<component name=\"[set ${d}(title)]\" category=\"[set ${d}(category)]\" optional=\"false\">"

    # OVERVIEW
    puts $id "  <overview>"
    puts $id "    <authors>"
    if {[info exists ${d}(authors)]} {
      foreach name [set ${d}(authors)] { 
        puts $id "      <author>$name</author>"
      }
    } else {
	createSciDialog -title "Module Creation Error" -error \
	    -message "Please provide at least one Author name.\n(Description Tab->Add button.)"
                return
    }    
    puts $id "    </authors>"

    # SUMMARY
    puts $id "    <summary>"
    puts $id "      [set ${d}(summary)]"
    puts $id "    </summary>"

    # DESCRIPTION
    puts $id "    <description>"
    puts $id "      <p>"
    puts $id "        [set ${d}(descript)]" 
    puts $id "      </p>"
    puts $id "    </description>"

    puts $id "  </overview>"

    # IO
    puts $id "  <io>"
    if { [info exists ${d}(dynamicport)] && [set ${d}(dynamicport)] } {
      puts $id "    <inputs lastportdynamic=\"yes\">"
    } else {
      puts $id "    <inputs lastportdynamic=\"no\">"
    }

    if {[info exists ${d}(iports)]} {

        foreach port [set ${d}(iports)] {
            global $port

            if { ![info exists ${port}(name)] || \
                     ![llength [set ${port}(name)]] || \
                     ![info exists ${port}(datatype)] || \
                     ![llength [set ${port}(datatype)]] } {
                createSciDialog -title "Module Creation Error" -error \
                    -message "Please provide all the input ports with names and datatypes.\n(Right click on the port on the module icon.)"
                return
            }
            puts $id "      <port>"
            puts $id "        <name>[set ${port}(name)]</name>"
            puts $id "        <datatype>[set ${port}(datatype)]</datatype>"
            puts $id "      </port>"
        }
    }
    puts $id "    </inputs>"

    if {[info exists ${d}(oports)]} {
      puts $id "    <outputs>"
      foreach port [set ${d}(oports)] {
        global $port
	if {![info exists ${port}(name)] || \
                ![llength [set ${port}(name)]] || \
                ![info exists ${port}(datatype)] || \
                ![llength [set ${port}(datatype)]]} {
              createSciDialog -title "Module Creation Error" -error \
                      -message "Please provide all the output ports with names and datatypes.\n(Right click on the port on the module icon.)"
              return
	}
	puts $id "      <port>"
        puts $id "        <name>[set ${port}(name)]</name>"
        puts $id "        <datatype>[set ${port}(datatype)]</datatype>"
        puts $id "      </port>"
      }
      puts $id "    </outputs>"
    }
    puts $id "  </io>"

    # GUI
    if {[set ${d}(hasgui)]} {
        puts $id "  <gui>"
        puts $id "    <description>"
        puts $id "      <p>"
        puts $id "        gui exists" 
        puts $id "      </p>"
        puts $id "    </description>"
        puts $id "  </gui>"
    }

    puts $id "</component>"
    close $id

    CreateNewModule [set ${d}(package)] [set ${d}(category)] [set ${d}(title)]
}

proc CreateNewModule { packname catname compname } {

    set xmlname "cwmmtemp.xml"

    set psepath "[netedit getenv SCIRUN_SRCDIR]"

    if { $compname=="" || $packname=="" || $catname==""} {
	createSciDialog -title "ERROR" -error \
                        -message "One or more of the entries was left blank.\nAll entries must be filled in."
	return
    }

    if {![file isdirectory $psepath]} {
	createSciDialog -title "PATH TO SCIRUN ERROR" -error \
	                -message "The path \"$psepath\" is already in use by a non-directory file."
	return
    }
    
    set basepath $psepath/Packages/$packname
    if {$packname=="SCIRun"} {
	set basepath $psepath
    } else {
	if {![file exists $basepath]} {
	    set answer [createSciDialog -title "PACKAGE NAME WARNING" -warning \
		-message "Package \"$basepath\" does not exist.  Create it now? \n \
                     (If yes, the category \"$psepath/src/Packages/ \
		     $packname/$catname\"\
                     will also be created.)" -button1 "Yes" -button2 "No"]

            if { $answer == 1 } {
		netedit create_pac_cat_mod $psepath $packname $catname $compname $xmlname
                destroy .componentWizard
                newPackageMessage $packname
            }
	    return
	}

	if {![file isdirectory $basepath]} {
	    createSciDialog -title "PACKAGE NAME ERROR" -error \
                    -message "The name \"$basepath\" is already in use by a non-package file."
	    return
	}
    }
#	   [file exists $basepath/sub.mk] &&
#          ![file isdirectory $basepath/sub.mk] &&

    if {![expr \
           [file exists $basepath/Dataflow] &&\
	   [file isdirectory $basepath/Dataflow] &&\
           [file exists $basepath/Dataflow/Modules] && \
           [file isdirectory $basepath/Dataflow/Modules] && \
           [file exists $basepath/Dataflow/XML] && \
           [file isdirectory $basepath/Dataflow/XML]]} {
       createSciDialog -title "PACKAGE ERROR" -error \
                      -message "The file \"$basepath\" does not appear\
                       to be a valid package or is somehow corrupt.\
                       The module \"$compname\" will not be added.\n\n\
                       See the \"Create A New Module\" documentation for\
                       more information."
	return
    }
             
    if {![file exists $basepath/Dataflow/Modules/$catname]} {
        set answer [createSciDialog -title "CATEGORY NAME WARNING" -warning \
                    -message "Category \"$basepath/Dataflow/Modules/$catname\"\
		    does not exist.  Create it now?" -button1 "Yes" -button2 "No"]
        if { $answer == 1 } {
            netedit create_cat_mod $psepath $packname $catname $compname $xmlname
            destroy .componentWizard; 
            newModuleMessage $compname
        }
	return
    }

    if {![file isdirectory \
	    $basepath/Dataflow/Modules/$catname]} {
	createSciDialog -title "CATEGORY NAME ERROR" -error \
                      -message "The name \"$basepath/Dataflow/Modules/$catname\"\
                       is already in use by a non-category file."
	return	
    }

    if {![file exists \
	    $basepath/Dataflow/Modules/$catname/sub.mk]} {
	createSciDialog -title "CATEGORY ERROR" -error \
                      -message "The file \"$basepath/Dataflow/Modules/$catname\"\
                       does not appear to be a valid category or is\
                       somehow corrupt.  The Module \"$compname\" will\
                       not be added.\n\n\
                       See the \"Create A New Module\" documentation for\
                       more information."
	return
    }

    if {[file exists \
	    $basepath/Dataflow/Modules/$catname/$compname.cc]} {
	createSciDialog -title "MODULE NAME ERROR" -error \
		        -message "The name \"$basepath/Dataflow/Modules/$catname/$compname\"\
                                  is already in use by another file."
	return
    }

    netedit create_mod $psepath $packname $catname $compname $xmlname
    destroy .componentWizard
    
    newModuleMessage $compname
}

proc newPackageMessage {pac} {
    createSciDialog -title "FINISHED CREATING NEW MODULE"\
            -message [join [concat {"In order to use the newly created package"} \
                                   {"you will have to quit SCIRun,"} \
                                   {"reconfigure (i.e. configure --enable-package=\"$pac\"),"} \
                                   {"and rebuild the PSE (gmake)."}] \n]
}

proc newModuleMessage {mod} {
    createSciDialog -title "FINISHED CREATING NEW MODULE" -warning \
	          -message "\nIn order to use the new module \"$mod\",\nyou must quit SCIRun, and then rebuild the PSE (gmake)."
}


