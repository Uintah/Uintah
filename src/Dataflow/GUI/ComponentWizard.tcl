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



#package require Iwidgets 3.0

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
    wm minsize .componentWizard 470 600

    # Tab panel
    iwidgets::tabnotebook $w.tabs 
    pack $w.tabs -padx $PADi -pady $PADi -fill both -expand yes

    # Horizontal separator
    frame $w.separator -height 2 -relief sunken -borderwidth 2
    pack  $w.separator -fill y -padx 5 -pady 5 -expand no -fill x

    # Close / Create buttons
    frame $w.buttons
    pack $w.buttons -ipadx $PADi -ipady $PADi -fill x -expand no

#    button $w.buttons.open -text "Open" 
#    pack $w.buttons.open -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

#    button $w.buttons.save -text "Save" -command "array set $d \[array get $tmpd\]"
#    pack $w.buttons.save -padx $PADi -ipadx $PADi -ipady $PADi -expand no -side left

    button $w.buttons.close -text "Close" -command "wm withdraw $w"
    button $w.buttons.create -text "Create" -command "array set $d \[array get $tmpd\]; generateXML $d"

    pack $w.buttons.create $w.buttons.close -side right -expand yes -fill x -padx $PADi

    set io_gui [$w.tabs add -label "I/O"]
    make_io_gui_pane $io_gui $tmpd

    set overview [$w.tabs add -label "Description"]
    make_description_pane $overview $tmpd

    # For now, under the 'simple is better' ideology, turn off the Implementation and Testing panes.
    #set implementation [$w.tabs add -label "Implementation"]
    #make_implementation_pane $implementation $tmpd
    #
    #set testing [$w.tabs add -label "Testing"]
    #make_testing_pane $testing $tmpd
    
    $w.tabs view "I/O"

    moveToCursor $w "leave_up"
}

proc make_io_gui_pane {p d} {
    set PAD .05
    set PADi [concat $PAD i]
    global $d


    # Create the main frames  (Canvas Frame (cf), Command Frame (cmds), Check Button Frame (cbf))
    frame $p.cf
    frame $p.cmds -border 1 -relief groove
    frame $p.cbf  -border 1 -relief groove

    # Create the canvas
    canvas $p.cf.c -relief sunken -borderwidth 3 -background #036
    pack $p.cf.c -expand y -fill both

    if { ![info exists ${d}(hasgui)] } {
        set ${d}(hasgui) 0
    }
    if { ![info exists ${d}(dynamicport)] } {
        set ${d}(dynamicport) 0
    }

    checkbutton $p.cbf.hasgui -text "Has GUI" -variable ${d}(hasgui) \
        -command "eval gui $p.cf.c \[set ${d}(hasgui)\]"
    global ${d}.dynamicport
    checkbutton $p.cbf.dynamicport -text "Last port is dynamic" -variable ${d}(dynamicport)

    pack $p.cbf.dynamicport $p.cbf.hasgui -side left -fill x -expand yes

    make_icon $p.cf.c 1.5i 1.375i [set ${d}(hasgui)] $d

    set modframe $p.cf.c.moduleFakeModule
    button $p.cmds.add_iport -text "Add Input Port"  -command "eval add_port $modframe i $d"
    button $p.cmds.add_oport -text "Add Output Port" -command "eval add_port $modframe o $d"

    #button $p.cmds.add_ifile -text "Add Input File"   -command ""
    #button $p.cmds.add_ofile -text "Add Output File"  -command ""
    #button $p.cmds.add_idev -text "Add Input Device"  -command ""
    #button $p.cmds.add_odev -text "Add Output Device" -command ""
    
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

#    set guidescript $p.guidescript
#    create_text_entry $guidescript "Description:" $d guidescript
    frame $p.cp -bd 1 -relief sunken

    frame $p.cp.name
    label $p.cp.name.label -text "Module Name: " -width 20 -anchor e
    entry $p.cp.name.entry -textvar ${d}(title) -width 30

    frame $p.cp.pack
    label $p.cp.pack.label -text "Package: " -width 20 -anchor e
    entry $p.cp.pack.entry -textvar ${d}(package) -width 30

    frame $p.cp.cat
    label $p.cp.cat.label -text "Category: " -width 20 -anchor e
    entry $p.cp.cat.entry -textvar ${d}(category) -width 30

    frame $p.cp.path
    label $p.cp.path.label -text "Path to SCIRun: " -width 20 -anchor e
    entry $p.cp.path.entry -textvar ${d}(path) -width 30

    pack $p.cmds.add_iport -padx $PADi -pady $PADi -ipady $PADi -expand no -side top -fill x
    pack $p.cmds.add_oport -padx $PADi -pady $PADi -ipady $PADi -expand no -side top -fill x

    pack $p.cp.name $p.cp.pack $p.cp.cat $p.cp.path -side top -pady $PADi

    pack $p.cp.name.label $p.cp.name.entry -side left
    pack $p.cp.pack.label $p.cp.pack.entry -side left
    pack $p.cp.cat.label $p.cp.cat.entry -side left
    pack $p.cp.path.label $p.cp.path.entry -side left
    
    trace variable ${d}(title) w "update_title_entry_bind"

    #pack $p.cmds.add_ifile -padx $PADi -pady $PADi -ipady $PADi -expand yes -side top -anchor nw -fill x
    #pack $p.cmds.add_ofile -padx $PADi -pady $PADi -ipady $PADi -expand yes -side top -anchor nw -fill x
    #pack $p.cmds.add_idev -padx $PADi -pady $PADi  -ipady $PADi -expand yes -side top -anchor nw -fill x
    #pack $p.cmds.add_odev -padx $PADi -pady $PADi  -ipady $PADi -expand yes -side top -anchor nw -fill x

    ### Packing:
    #    .cf (Canvas Frame) | .cmds (Command Frame)
    #    .cbf (Check Buttons Frame)
    #    .cp (Name entries)
    #
    grid $p.cf   -sticky news -padx 5 -pady 2
    grid $p.cmds -column 1 -row 0 -sticky nws -pady 2
    grid $p.cbf  -sticky ew -padx 5 -ipadx 5 -ipady 2
    grid $p.cp     -columnspan 2 -pady 0 -padx 5 -sticky ew

#    pack $guidescript -fill x -side bottom -anchor sw \
#        -padx $PADi 
}


proc update_title_entry_bind {} {
    global .componentWizard.tmpdata

    set p [.componentWizard.tabs childsite "I/O"]
    set title_pentry $p.cf.c.moduleFakeModule.ff.title
    set tmp [set .componentWizard.tmpdata(title)]

    puts "here: title is $tmp"

    set_prompted_entry $title_pentry $tmp
}


proc make_description_pane {p d} {
    global $d

    set authors $p.authors
    create_clb_entry $authors "Authors:" "<enter new author here>" $d authors

    set summary $p.summary
    create_text_entry $summary "Summary:" $d summary

    set uiinfo $p.uiinfo
    create_text_entry $uiinfo "GUI Info:" $d uiinfo

    set descript $p.descript
    create_text_entry $descript "Description:" $d descript

    set lexamplesr $p.lexamplesr
    label $lexamplesr -text "Example network: "

    set eexamplesr $p.eexamplesr
    prompted_entry $eexamplesr "<enter example network filename>" "
            global $d;
            set ${d}(examplesr) \[get_prompted_entry $eexamplesr\];
        " 
    if [info exists ${d}(examplesr)] {
        set_prompted_entry $p.eexamplesr [set ${d}(examplesr)]
    }

    pack $authors  -side top -fill both -expand true -padx .1c -pady .1c
    pack $summary  -side top -anchor w -fill x -expand true -padx .1c -pady .1c
    pack $descript -side top -anchor w -fill x -expand true -padx .1c -pady .1c
    pack $uiinfo   -side top -anchor w -fill x -expand true -padx .1c -pady .1c

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
    text $t -wrap word -yscrollcommand "$sy set" -height 5 -width 60
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
    
    $canvas create window $modx $mody -window $modframe \
	    -tags FakeModule -anchor center
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

    # DELETE The following lines:
    #qwerty
    #$menu add command -label "Edit" -command "edit_port $portnum"
}

proc configPorts {icon type d} {
    set ports ${type}ports
    set i 0
    global $d

    puts "ports is $ports"

    foreach t [set ${d}($ports)] {

        puts "t is $t"

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
    wm minsize $w 470 600

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
        -wrap word -yscrollcommand "$sydescript set" -height 5 -width 50
    pack $edescript -side left -fill x -expand true
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
        

    set close $fbuttons.close
    global $close
    button $close -text Close -command "destroy $w"
    pack $close -side left -padx .1c -pady .1c -ipadx .1c -ipady .1c

    pack $f -fill both -expand yes -side top
    pack $fdescript -fill both -expand yes -side top
    pack $fcompnames -fill both -expand yes -side top
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
    if { ![info exists ${d}(title)] || \
             ![llength [set ${d}(title)]] || \
             ![info exists ${d}(category)] || \
             ![llength [set ${d}(category)]] || \
             ![info exists ${d}(package)] || \
             ![llength [set ${d}(package)]] || \
             ![info exists ${d}(path)] || \
             ![llength [set ${d}(path)]] } {
                 
      createSciDialog -title "Module Creation Error" \
              -message [join [concat {"One or more of the following required fields is empty:"} \
              {"Module Name, Package Name, Category Name, Path to SCIRun"}] \n] -error
      return
    } 
    puts $id "<component name=\"[set ${d}(title)]\" category=\"[set ${d}(category)]\" optional=false>"
    puts $id "  <overview>"
    if {[info exists ${d}(authors)]} {
      puts $id "    <authors>"
      foreach name [set ${d}(authors)] { 
        puts $id "      <author>$name</author>"
      }
      puts $id "    </authors>"
    }
    puts $id "    <summary>"
    puts $id "    </summary>"
    puts $id "    <description>"
    puts $id "    </description>"
    if {[info exists ${d}(examplesr)]} {
      if {[llength [set ${d}(examplesr)]]} {
        puts $id "    <examplesr>"
        puts $id "      [set ${d}(examplesr)]"
        puts $id "    </examplesr>"
      }
    }
    puts $id "  </overview>"
    puts $id "  <implementation>"
    puts $id "  </implementation>"
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
    puts $id "    <outputs>"
    if {[info exists ${d}(oports)]} {
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
    }
    puts $id "    </outputs>"
    puts $id "  </io>"
    if {[set ${d}(hasgui)]} {
      puts $id "  <gui>"
      puts $id "    <parameter>"
      puts $id "      <widget>Label</widget>"
      puts $id "      <label>Autogenerated GUI explanation</label>"
      puts $id "    </parameter>"
      puts $id "  </gui>"
    }
    puts $id "  <testing>"
    puts $id "  </testing>"
    puts $id "</component>"
    close $id

    CreateNewModule [set ${d}(package)] [set ${d}(category)] [set ${d}(path)] \
	            [set ${d}(title)]
}

proc CreateNewModule { packname catname psepath compname } {
#    netedit load_component_spec cwmmtemp.xml $package $category $path

    set xmlname "cwmmtemp.xml"

    if {$psepath=="" || $compname=="" || $packname=="" || $catname==""} {
	createSciDialog -title "ERROR" -error \
                        -message "One or more of the entries was left blank.\nAll entries must be filled in."
	return
    }

    if {![file exists $psepath]} {
        createSciDialog -title "PATH TO SCIRUN ERROR" -error \
	                -message "The path \"$psepath\" does not exist.  Please enter the correct path."
	return
    }

    if {![file isdirectory $psepath]} {
	createSciDialog -title "PATH TO SCIRUN ERROR" -error \
	                -message "The path \"$psepath\" is already in use by a non-directory file."
	return
    }
    
    set basepath $psepath/src/Packages/$packname
    if {$packname=="SCIRun"} {
	set basepath $psepath/src
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
	   [file exists $basepath/Core] &&\
	   [file isdirectory $basepath/Core] &&\
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
        if { $anwer == 1 } {
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


