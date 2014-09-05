#puts "NetworkEditor.tcl start"

source $PSECoreTCL/defaults.tcl
source $PSECoreTCL/devices.tcl

set modname_font "-Adobe-Helvetica-Bold-R-Normal-*-12-120-75-*"
set ui_font "-Adobe-Helvetica-Medium-R-Normal-*-12-120-75-*"
set time_font "-Adobe-Courier-Medium-R-Normal-*-12-120-75-*"

set mainCanvasWidth    4500.0
set mainCanvasHeight   4500.0
set miniCanvasWidth     150.0
set miniCanvasHeight    150.0
set SCALEX [expr $mainCanvasWidth/$miniCanvasWidth]
set SCALEY [expr $mainCanvasHeight/$miniCanvasHeight]

# Records where the mouse was pressed to bring up the Modules Menu,
# thus allowing the module to be create at (or near) that location.
set mouseX 0
set mouseY 0

global maincanvas
set maincanvas ".bot.neteditFrame.canvas"
global minicanvs
set minicanvas ".top.globalViewFrame.canvas"

global loading
set loading 0


proc resource {} {
}

proc makeNetworkEditor {} {

    wm minsize . 100 100
    wm geometry . 800x800+0+0

    frame .main_menu -relief raised -borderwidth 3
    pack .main_menu -fill x
    
    menubutton .main_menu.file -text "File" -underline 0 \
	-menu .main_menu.file.menu
    menu .main_menu.file.menu -tearoff false
    menu .main_menu.file.menu.new -tearoff false
    .main_menu.file.menu.new add command -label "Module..." \
        -underline 0 -command "CreateNewModule"
    .main_menu.file.menu add command -label "Save..." -underline 0 \
	-command "popupSaveMenu"
    .main_menu.file.menu add command -label "Load..." -underline 0 \
	-command "popupLoadMenu"
    .main_menu.file.menu add cascade -label "New" -underline 0\
        -menu .main_menu.file.menu.new

# This was added by Mohamed Dekhil to add some infor to the net
    .main_menu.file.menu add command -label "Add Info..." -underline 0 \
	-command "popupInfoMenu"

    .main_menu.file.menu add command -label "Quit" -underline 0 \
	    -command "netedit quit"


    menubutton .main_menu.stats -text "Statistics" -underline 0 \
	-menu .main_menu.stats.menu
    menu .main_menu.stats.menu
    .main_menu.stats.menu add command -label "Debug..." -underline 0 \
	    -command showDebugSettings
    .main_menu.stats.menu add command -label "Memory..." -underline 0 \
	    -command showMemStats
    .main_menu.stats.menu add command -label "Threads..." -underline 0 \
	    -command showThreadStats

    menubutton .main_menu.help -text "Help" -underline 0 \
	-menu .main_menu.help.menu
    menu .main_menu.help.menu

    .main_menu.help.menu add command -label "Help..." -underline 0 \
	    -command { tk_messageBox -message "I'm helpless" }

#    pack .main_menu.file        \
#         .main_menu.modules     \
#         .main_menu.appModules     -side left
    pack .main_menu.file -side left
    pack .main_menu.help        \
         .main_menu.stats          -side right

#    tk_menuBar .main_menu .main_menu.file .main_menu.modules \
#                          .main_menu.stats .main_menu.help
    tk_menuBar .main_menu .main_menu.file .main_menu.stats .main_menu.help

    frame .top -borderwidth 5
    pack  .top -side top -fill x
    frame .bot -borderwidth 5
    pack  .bot -side bottom -expand yes -fill both

    frame .top.globalViewFrame -relief sunken -borderwidth 3
    frame .bot.neteditFrame -relief sunken -borderwidth 3

    global mainCanvasHeight mainCanvasWidth
    canvas .bot.neteditFrame.canvas \
        -scrollregion "0 0 $mainCanvasWidth $mainCanvasHeight" \
	-bg #036


    # bgRect is just a rectangle drawn on the neteditFrame Canvas
    # so that the Modules List Menu can be bound to it using mouse
    # button 3.  The Modules List Menu can't be bound to the canvas
    # itself because mouse events are sent to both the objects on the
    # canvas (such as the lines connection the modules) and the canvas.

    set bgRect [.bot.neteditFrame.canvas create rectangle 0 0 \
	                         $mainCanvasWidth $mainCanvasWidth -fill #036]

    menu .bot.neteditFrame.canvas.modulesMenu -tearoff false

    scrollbar .bot.neteditFrame.hscroll -relief sunken \
	    -orient horizontal \
	    -command ".bot.neteditFrame.canvas xview"
    scrollbar .bot.neteditFrame.vscroll -relief sunken \
	-command ".bot.neteditFrame.canvas yview" 

    pack .bot.neteditFrame -expand yes -fill both -padx 4

    grid .bot.neteditFrame.canvas .bot.neteditFrame.vscroll
    grid .bot.neteditFrame.hscroll

    grid columnconfigure .bot.neteditFrame 0 -weight 1 
    grid rowconfigure    .bot.neteditFrame 0 -weight 1 

    grid config .bot.neteditFrame.canvas -column 0 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "snew" 
    grid config .bot.neteditFrame.hscroll -column 0 -row 1 \
	    -columnspan 1 -rowspan 1 -sticky "ew" -pady 2
    grid config .bot.neteditFrame.vscroll -column 1 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "sn" -padx 2

    # Create Error Message Window...
    frame .top.errorFrame -borderwidth 3 
    text .top.errorFrame.text -relief sunken -bd 3 -bg #036 -fg white \
	    -yscrollcommand ".top.errorFrame.s set" -height 10 -width 180 
    .top.errorFrame.text insert end "Messages:\n"
    .top.errorFrame.text insert end "--------\n\n"
    .top.errorFrame.text tag configure errtag -foreground red

# Why on earth was this here?
#    .top.errorFrame.text configure -state disabled

    scrollbar .top.errorFrame.s -relief sunken \
	    -command ".top.errorFrame.text yview"
    pack .top.errorFrame.s -side right -fill y -padx 4
    pack .top.errorFrame.text -expand yes -fill both
    global netedit_errortext
    set netedit_errortext .top.errorFrame.text

    pack .top.globalViewFrame -side left -padx 4
    pack .top.errorFrame -side right -fill both -expand yes

    global miniCanvasHeight miniCanvasWidth
    canvas .top.globalViewFrame.canvas \
	-bg #036 -width $miniCanvasWidth -height $miniCanvasHeight
    pack   .top.globalViewFrame.canvas 

    createCategoryMenu

    global netedit_canvas
    global netedit_mini_canvas
    set netedit_canvas .bot.neteditFrame.canvas
    set netedit_mini_canvas .top.globalViewFrame.canvas

    set viewAreaBox \
      [ $netedit_mini_canvas create rectangle 0 0 1 1 -outline black ]

    .bot.neteditFrame.canvas configure \
	-xscrollcommand "updateCanvasX $viewAreaBox" \
        -yscrollcommand "updateCanvasY $viewAreaBox"

    bind $netedit_mini_canvas <B1-Motion> \
      "updateCanvases $netedit_mini_canvas $netedit_canvas $viewAreaBox %x %y"
    bind $netedit_mini_canvas <1> \
      "updateCanvases $netedit_mini_canvas $netedit_canvas $viewAreaBox %x %y"
    bind $netedit_canvas <Configure> \
      "handleResize $netedit_mini_canvas $netedit_canvas $viewAreaBox %w %h"
    $netedit_canvas bind $bgRect <ButtonPress-3> "modulesMenuPressCB %x %y"
    bind . <KeyPress-Down>  { $netedit_canvas yview moveto [expr [lindex \
	    [$netedit_canvas yview] 0] + 0.01 ] }
    bind . <KeyPress-Up>    { $netedit_canvas yview moveto [expr [lindex \
	    [$netedit_canvas yview] 0] - 0.01 ] }
    bind . <KeyPress-Left>  { $netedit_canvas xview moveto [expr [lindex \
	    [$netedit_canvas xview] 0] - 0.01 ] }
    bind . <KeyPress-Right> { $netedit_canvas xview moveto [expr [lindex \
	    [$netedit_canvas xview] 0] + 0.01 ] }
}

proc modulesMenuPressCB { x y } {

    set canvas .bot.neteditFrame.canvas

    global mouseX mouseY
    set mouseX $x
    set mouseY $y
    tk_popup $canvas.modulesMenu [expr $x + [winfo rootx $canvas]] \
	                         [expr $y + [winfo rooty $canvas]]

}

proc handleResize { minicanv maincanv box w h } {
    global SCALEX SCALEY

    set ulx  [lindex [$minicanv coords $box] 0]
    set uly  [lindex [$minicanv coords $box] 1]

    set wid [ expr [ winfo width $maincanv ] / $SCALEX ]
    set hei [ expr [ winfo height $maincanv ] / $SCALEY ]

    $minicanv coords $box $ulx $uly [expr $ulx + $wid] [expr $uly + $hei]
}

proc updateCanvasX { box beg end } {
    global SCALEX SCALEY
    global netedit_canvas netedit_mini_canvas
    global miniCanvasWidth miniCanvasHeight

    # Tell the scroll bar to upate

    .bot.neteditFrame.hscroll set $beg $end

    # Update the view area box 

    set wid [expr [winfo width $netedit_canvas] / $SCALEX]

    set uly [lindex [$netedit_mini_canvas coords $box] 1]
    set lry [lindex [$netedit_mini_canvas coords $box] 3]
    set ulx [ expr $beg * $miniCanvasWidth ]
    set lrx [ expr $ulx + $wid - 1 ]

    $netedit_mini_canvas coords $box $ulx $uly $lrx $lry
}

proc updateCanvasY { box beg end } {
    global SCALEX SCALEY
    global netedit_canvas netedit_mini_canvas
    global miniCanvasWidth miniCanvasHeight

    # Tell the scroll bar to upate

    .bot.neteditFrame.vscroll set $beg $end

    # Update the view area box 

    set hei [ expr [ winfo height $netedit_canvas ] / $SCALEY ]

    set ulx [lindex [$netedit_mini_canvas coords $box] 0]
    set uly [ expr $beg * $miniCanvasHeight ]
    set lrx [lindex [$netedit_mini_canvas coords $box] 2]
    set lry [ expr $uly + $hei - 1 ]

    $netedit_mini_canvas coords $box $ulx $uly $lrx $lry
}

proc updateCanvases { minicanv maincanv box x y } {

    global miniCanvasWidth miniCanvasHeight

    # Find the width and height of the mini box.

    set wid [expr [lindex [$minicanv coords $box] 2] - \
	          [lindex [$minicanv coords $box] 0] ]
    set hei [expr [lindex [$minicanv coords $box] 3] - \
	          [lindex [$minicanv coords $box] 1] ]

    if [expr $x < ($wid / 2)] { set x [expr $wid / 2] }
    if [expr $x > ($miniCanvasWidth - ($wid / 2))] \
         { set x [ expr $miniCanvasWidth - ($wid / 2) - 1 ] }
    if [expr $y < ($hei / 2)] { set y [expr $hei / 2] }
    if [expr $y > ($miniCanvasHeight - ($hei / 2))] \
         { set y [ expr $miniCanvasHeight - ($hei / 2) - 1 ] }

    # Move the minibox to the new location

    $minicanv coords $box [expr $x - ($wid/2)] [expr $y - ($hei/2)] \
	                  [expr $x + ($wid/2)] [expr $y + ($hei/2)]

    # Update the region displayed in the main canvas.
    # The scroll bars seem to automagically update.
    $maincanv xview moveto [expr [expr $x - $wid/2] / $miniCanvasWidth ]
    $maincanv yview moveto [expr [expr $y - $hei/2] / $miniCanvasHeight ]
}

# All this while loop does is remove spaces from the string.  There's
# got to be a better way.  I'm really bad at TCL.

proc removeSpaces { str } {
  while {[string first " " $str] != -1} {
    set n [string first " " $str]
    set before [string range $str 0 [expr $n-1]]
    set after [string range $str [expr $n+1] \
              [string length $str] ]
    set str "${before}$after"
  }
  return $str
}

proc createCategoryMenu {} {

#  puts "Building Module Menus..."

  foreach package [netedit packageNames] {
    set packageToken [removeSpaces "menu_$package"]
#    puts "  $package -> $packageToken"

    # Add the cascade button and menu for the package to the menu bar

    menubutton .main_menu.$packageToken -text "$package" -underline 0 \
      -menu .main_menu.$packageToken.menu
    menu .main_menu.$packageToken.menu
    pack .main_menu.$packageToken -side left

    # Add a separator to the right-button menu for this package if this
    # isn't the first package to go in there

    if { [.bot.neteditFrame.canvas.modulesMenu index end] != "none" } \
      { .bot.neteditFrame.canvas.modulesMenu add separator }

    foreach category [netedit categoryNames $package] {
      set categoryToken [removeSpaces "menu_${package}_$category"]
#      puts "    $category -> $categoryToken"

      # Add the category to the menu bar menu

      .main_menu.$packageToken.menu add cascade -label "$category" \
        -menu .main_menu.$packageToken.menu.m_$categoryToken
      menu .main_menu.$packageToken.menu.m_$categoryToken -tearoff false

      # Add the category to the right-button menu

      .bot.neteditFrame.canvas.modulesMenu add cascade -label "$category" \
        -menu .bot.neteditFrame.canvas.modulesMenu.m_$categoryToken
      menu .bot.neteditFrame.canvas.modulesMenu.m_$categoryToken -tearoff false

      foreach module [netedit moduleNames $package $category] {
        set moduleToken [removeSpaces $module]
#        puts "      $module -> $moduleToken"

        # Add a button for each module to the menu bar category menu and the
        # right-button menu

        .main_menu.$packageToken.menu.m_$categoryToken add command \
          -label "$module" \
          -command "addModule \"$package\" \"$category\" \"$module\""
        .bot.neteditFrame.canvas.modulesMenu.m_$categoryToken add command \
          -label "$module" \
          -command "addModuleAtMouse \"$package\" \"$category\" \"$module\""
      }
    }
  }

#    set cats [netedit packages]
#
#    foreach cat $cats {
#	# The first item in the list is the category name
#        set package [lindex $cat 0]
#	set name [lindex $cat 1]
#	set mods [lindex $cat 2]
#	set module_cats($name) $mods
#
#puts $package;
#puts $name;
#puts $mods;
#
#	.main_menu.modules.menu add cascade -label $name \
#		-menu .main_menu.modules.menu.m_$name
#	.bot.neteditFrame.canvas.modulesMenu add cascade -label $name \
#		-menu .bot.neteditFrame.canvas.modulesMenu.m_$name
#
#	menu .main_menu.modules.menu.m_$name -tearoff false
#	menu .bot.neteditFrame.canvas.modulesMenu.m_$name -tearoff false
#
#	foreach mod $mods {
#	    .main_menu.modules.menu.m_$name add command -label $mod \
#		    -command "addModule $mod"
#
#	    .bot.neteditFrame.canvas.modulesMenu.m_$name add command \
#		    -label $mod \
#		    -command "addModuleAtMouse $mod"
#	}
#    }

#    # Handle Application Specific Modules
#
#    set cats              [netedit catlist application]
#    set uintahModulesMenu .main_menu.appModules.menu
#
#    .bot.neteditFrame.canvas.modulesMenu add separator
#
#    foreach cat $cats {
#	# The first item in the list is the category name
#	set name [lindex $cat 0]
#	set mods [lindex $cat 1]
#	set module_cats($name) $mods
#
#	$uintahModulesMenu add cascade -label $name \
#		-menu $uintahModulesMenu.m_app_$name
#	.bot.neteditFrame.canvas.modulesMenu add cascade -label $name \
#		-menu .bot.neteditFrame.canvas.modulesMenu.m_app_$name
#
#	menu $uintahModulesMenu.m_app_$name -tearoff false
#	menu .bot.neteditFrame.canvas.modulesMenu.m_app_$name -tearoff false
#
#	foreach mod $mods {
#	    $uintahModulesMenu.m_app_$name add command -label $mod \
#		    -command "addModule $mod"
#
#	    .bot.neteditFrame.canvas.modulesMenu.m_app_$name add command \
#		    -label $mod \
#		    -command "addModuleAtMouse $mod"
#	}
#    }

}

proc moveModule {name} {
    
}
##########################
proc addModule { package category module } {
    return [addModuleAtPosition "$package" "$category" "$module" 10 10]
}

proc addModuleAtMouse { package category module } {
    global mouseX mouseY

    return [ addModuleAtPosition "$package" "$category" "$module" $mouseX \
             $mouseY ]
}

proc addModuleAtPosition {package category module xpos ypos} {
    global mainCanvasWidth mainCanvasHeight
    global loading

    set mainCanvasWidth 4500
    set mainCanvasHeight 4500
    
    # create the modules at their relative positions only when not loading from a script.
    
	set xpos [expr $xpos+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*$mainCanvasWidth))])]
	set ypos [expr $ypos+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*$mainCanvasHeight))])]
    
    set modid [netedit addmodule "$package" "$category" "$module"]
    # Create the itcl object
    set className [removeSpaces "${package}_${category}_${module}"]
    if {[catch "$className $modid" exception]} {
	# Use generic module
	if {$exception != "invalid command name \"$className\""} {
	    bgerror "Error instantiating iTcl class for module:\n$exception";
	}
	Module $modid -name "$module"
    }
    $modid make_icon .bot.neteditFrame.canvas \
	             .top.globalViewFrame.canvas $xpos $ypos
    update idletasks
    return $modid
}

proc addConnection {omodid owhich imodid iwhich} {
    set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
    set portcolor [lindex [lindex [$omodid-c oportinfo] $owhich] 0]
    
    global connection_list
    set connection_list "$connection_list {$omodid $owhich $imodid $iwhich\
	    $portcolor}"
    
    buildConnection $connid $portcolor $omodid $owhich $imodid $iwhich
    configureOPorts $omodid
    configureIPorts $imodid
    update idletasks
}

# Utility procedures to support dragging of items.

proc itemStartDrag {c x y} {
    global lastX lastY
    set lastX [$c canvasx $x]
    set lastY [$c canvasy $y]
}

proc itemDrag {c x y} {
    global lastX lastY
    set x [$c canvasx $x]
    set y [$c canvasy $y]
    $c move current [expr $x-$lastX] [expr $y-$lastY]
    set lastX $x
    set lastY $y
}

proc popupSaveMenu {} {
    set types {
	{{Uintah Scripts Files} {.uin} }
	{{SCIRun Script} {.sr} }
	{{Other} { * } }
    } 
    set netedit_savefile [ tk_getSaveFile -defaultextension {.uin} \
			       -filetypes $types ]

    if { $netedit_savefile != "" } {
	saveMacroModules
	netedit savenetwork  $netedit_savefile
    }
}

proc popupLoadMenu {} {
    global netedit
    set types {
	{{Uintah Scripts Files} {.uin} }
	{{SCIRun Script} {.sr} }
	{{Other} { * } }
    } 
    
    set netedit_loadfile [tk_getOpenFile -filetypes $types ]
    
    if { [file exists $netedit_loadfile] } {
	loadfile $netedit_loadfile
    }
}

proc CreateNewModule {} {
    set types {
	{ {Module Specification Files} {.xml} }
	{ {Other} { * } }
    }
    set specfile [tk_getOpenFile -filetypes $types]
    if { [file exists $specfile] } {
	netedit load_component_spec $specfile
    }
}

proc GetPathAndPackage { compaddr compname catname } {
    set w .newmoduledialog
    if {[winfo exists $w]} {
	destroy $w
    }

    toplevel $w
    wm title $w "Specify new component location"

    frame $w.row1
    frame $w.row2
    frame $w.row3
    frame $w.row4
    pack $w.row1 $w.row2 $w.row3 $w.row4 -pady 5 -padx 5

    label $w.row1.compname -text "Component name: $compname"
    label $w.row1.catname -text "Category name: $catname"
    pack $w.row1.compname $w.row1.catname -side top -f both -e y

    label $w.row2.psepath_l -text "PSE to insert into:"
    entry $w.row2.psepath 
    button $w.row2.browse -text "Browse"\
	   -command "$w.row2.psepath delete 0 end;\
	   $w.row2.psepath insert 0 \[tk_getOpenFile -filetypes\
	                   {{ {PSE executables} {pse} }
		            { {Other} { * } }}\]"
    pack $w.row2.psepath_l $w.row2.psepath $w.row2.browse \
	 -side left -f both -e y \

    label $w.row3.packname_l -text "Package to insert into:"
    entry $w.row3.packname 
    pack $w.row3.packname_l $w.row3.packname -side left -f both -e y
    
    button $w.row4.ok -text "Ok" -command "CreateNewModuleOk $compaddr \
	   $compname $catname \[$w.row3.packname get\] \[$w.row2.psepath get\]"
    button $w.row4.cancel -text "Cancel" -command "destroy $w"
    pack $w.row4.ok $w.row4.cancel -side left -f both -e y
    focus $w.row2.psepath
#    grab set $w
    tkwait window $w
}

proc CreateNewModuleOk { compaddr compname catname packname psepath} {
    set w .newmoduledialog

    if {$psepath=="" || $compname=="" || $packname=="" || $catname==""} {
	messagedialog "ERROR"\
                      "One or more of the entries was left blank.\
                       All entries must be filled in."
	return
    }

    if {![file exists $psepath]} {
	messagedialog "The path \"$psepath\" does not exist. \
		       Please choose another path."
	return
    }

    if {![file isdirectory $psepath]} {
	messagedialog "The path \"$psepath\" is already in use\
		       by a non-directory file"
	return
    }

    if {![file exists $psepath/src/$packname]} {
	yesnodialog "PACKAGE NAME WARNING" \
                    "Package \"$psepath/src/$packname\" does not exist. \
                     Create it now? \n \
                     (If yes, the category \"$psepath/src/$packname/$catname\"\
                     will also be created.)" \
		    "netedit create_pac_cat_mod $psepath $packname\
		    $catname $compname $compaddr; destroy $w;\
                    newpackagemessage $packname"
	return
    }

    if {![file isdirectory $psepath/src/$packname]} {
	messagedialog "PACKAGE NAME ERROR" \
                      "The name \"$psepath/src/$packname\" is already in use\
                       by a non-package file"
	return
    }

    if {![expr [file exists $psepath/src/$packname/Modules] && \
               [file isdirectory $psepath/src/$packname/Modules] && \
               [file exists $psepath/src/$packname/XML] && \
               [file isdirectory $psepath/src/$packname/XML] && \
               [file exists $psepath/src/$packname/sub.mk] && \
               ![file isdirectory $psepath/src/$packname/sub.mk]]} {
	messagedialog "PACKAGE ERROR" \
                      "The file \"$psepath/src/$packname\" does not appear\
                       to be a valid package or is somehow corrupt.\
                       The module \"$compname\" will not be added.\n\n\
                       See the \"Create A New Module\" documentation for\
                       more information."
	return
    }
             
    if {![file exists $psepath/src/$packname/Modules/$catname]} {
	yesnodialog "CATEGORY NAME WARNING" \
                    "Category \"$psepath/src/$packname/Modules/$catname\"\
		    does not exist.  Create it now?" \
		    "netedit create_cat_mod $psepath $packname\
		    $catname $compname $compaddr; destroy $w;\
		    newmodulemessage $compname"
	return
    }

    if {![file isdirectory $psepath/src/$packname/Modules/$catname]} {
	messagedialog "CATEGORY NAME ERROR" \
                      "The name \"$psepath/src/$packname/Modules/$catname\"\
                       is already in use by a non-category file"
	return	
    }

    if {![file exists $psepath/src/$packname/Modules/$catname/sub.mk]} {
	messagedialog "CATEGORY ERROR" \
                      "The file \"$psepath/src/$packname/Modules/$catname\"\
                       does not appear to be a valid category or is\
                       somehow corrupt.  The Module \"$compname\" will\
                       not be added.\n\n\
                       See the \"Create A New Module\" documentation for\
                       more information."
	return
    }

    if {[file exists $psepath/src/$packname/Modules/$catname/$compname.cc]} {
	messagedialog "MODULE NAME ERROR" \
		      "The name \
		      \"$psepath/src/$packname/Modules/$catname/$compname\"\
                      is already in use by another file"
	return
    }

    netedit create_mod $psepath $packname $catname $compname $compaddr
    destroy $w
    newmodulemessage $compname
}

proc newpackagemessage {pac} {
    messagedialog "FINISHED CREATING NEW MODULE"\
                  "In order to use the newly created package \
                   you will first have to close, \
                   reconfigure (i.e. configure --enable-package=\"$pac\"), \
                   and rebuild the PSE."
}

proc newmodulemessage {mod} {
    messagedialog "FINISHED CREATING NEW MODULE"\
	          "In order to use \"$mod\" you will first have to\
                   close, and then rebuild, the PSE."
}
                   
proc CreateNewModuleCancel {} {
    destroy .newmoduledialog
}

proc messagedialog {title message} {
    set w .errordialog
    if {[winfo exists $w]} {
	destroy $w
    }

    toplevel $w 
    wm title $w $title
    text $w.message -width 60 -height 10 -wrap word -relief flat
    $w.message insert 0.1 $message
    button $w.ok -text "Ok" -command "destroy $w"
    pack $w.message $w.ok -side top -padx 5 -pady 5
    focus $w.ok
    grab set $w
    tkwait window $w
}

proc yesnodialog {title message command} {
    set w .yesnodialog
    if {[winfo exists $w]} {
	destroy $w
    }

    toplevel $w
    wm title $w $title
    frame $w.top
    frame $w.bot
    text $w.top.message -width 60 -height 10 -wrap word -relief flat
    $w.top.message insert 0.1 $message
    button $w.bot.yes -text "Yes" -command "$command;destroy $w"
    button $w.bot.no -text "no" -command "destroy $w"
    pack $w.top $w.bot
    pack $w.top.message -padx 5 -pady 5
    pack $w.bot.yes $w.bot.no -side left -padx 5 -pady 5 
    focus $w
    grab set $w
    tkwait window $w
}

# This proc was added by Mohamed Dekhil to save some info about the net

proc popupInfoMenu {} {

    global userName
    global runDate
    global runTime
    global notes

    global oldUserName
    global oldRunDate
    global oldRunTime
    global oldNotes

    set oldUserName ""
    set oldRunDate ""
    set oldRunTime ""
    set oldNotes ""

    if [info exists userName] {set oldUserName $userName}
    if [info exists runDate] {set oldRunDate $runDate}
    if [info exists runTime] {set oldRunTime $runTime}
    if [info exists notes] {set oldNotes $notes}    

    set w .netedit_info
    if {[winfo exists $w]} {
	raise $w
	return;
    }
    toplevel $w

    frame $w.fname
    label $w.fname.lname -text "User: " -padx 3 -pady 3
    entry $w.fname.ename -width 50 -relief sunken -bd 2 -textvariable userName


    frame $w.fdt
    label $w.fdt.ldate -text "Date: " -padx 3 -pady 3 
    entry $w.fdt.edate -width 20 -relief sunken -bd 2 -textvariable runDate
#    label $w.fdt.edate -text [exec date] -padx 3 -pady 3 -relief sunken

    label $w.fdt.ltime -text "Time: " -padx 5 -pady 3 
    entry $w.fdt.etime -width 10 -relief sunken -bd 2 -textvariable runTime

    frame $w.fnotes
    label $w.fnotes.lnotes -text "Notes " -padx 2 -pady 5 
    text $w.fnotes.tnotes -relief sunken -bd 2 -yscrollcommand "$w.fnotes.scroll set"
    scrollbar $w.fnotes.scroll -command "$w.fnotes.tnotes yview"
    if [info exists notes] {$w.fnotes.tnotes insert 1.0 $notes}

    
    frame $w.fbuttons 
    button $w.fbuttons.ok -text "Done" -command "infoOk $w"
    button $w.fbuttons.clear -text "Clear All" -command "infoClear $w"
    button $w.fbuttons.cancel -text "Cancel" -command "infoCancel $w"

    pack $w.fname $w.fdt $w.fnotes $w.fbuttons -side top -padx 1 -pady 1 -ipadx 2 -ipady 2 -fill x

    pack $w.fname.lname $w.fname.ename -side left

    pack $w.fdt.ldate $w.fdt.edate $w.fdt.ltime $w.fdt.etime -side left 

    pack $w.fnotes.lnotes $w.fnotes.tnotes -side left
    pack $w.fnotes.scroll -side right -fill y

    pack $w.fbuttons.ok $w.fbuttons.clear $w.fbuttons.cancel -side right -padx 5 -pady 5 -ipadx 3 -ipady 3
}

proc infoClear {w} {
    global userName
    global runDate
    global runTime
    global notes

    set userName ""
    set runDate ""
    set runTime ""
    set notes ""

    $w.fnotes.tnotes delete 1.0 end
#    destroy $w
}

proc infoOk {w} {
    global notes

    set notes [$w.fnotes.tnotes get 1.0 end]
    destroy $w
}

proc infoCancel {w} {
    global userName
    global runDate
    global runTime
    global notes

    global oldUserName
    global oldRunDate
    global oldRunTime
    global oldNotes

    set userName $oldUserName
    set runDate $oldRunDate
    set runTime $oldRunTime
    set notes $oldNotes

    destroy $w
} 

proc createAlias {fromPackage fromCategory fromModule toPackage toCategory toModule} {
    set fromClassName [removeSpaces "${fromPackage}_${fromCategory}_${fromModule}"]
    set toClassName [removeSpaces "${toPackage}_${toCategory}_${toModule}"]
    itcl_class $toClassName "inherit $fromClassName"
}

proc loadfile {netedit_loadfile} {
    global loading
    set loading 0
    set group_info [sourcefile $netedit_loadfile]
    set loading 1

    if { ! [string match $group_info ""] } {
	[loadMacroModules $group_info]
    }
}

proc sourcefile {netedit_loadfile} {
    # set loading to 1
    global loading
    set loading 1

    # Check to see of the file exists; exit if it doesn't
    if { ! [file exists $netedit_loadfile] } {
	puts "$netedit_loadfile: no such file"
	return
    }
    
    set fchannel [open $netedit_loadfile]
    
    set curr_line ""
    set stage 1

    global info_list
    set info_list ""

    # Used in tracking modnames
    set curr_modname ""
    set counter -1

    # read in the first line of the file
    set curr_line [gets $fchannel]
    
    set group_info ""
    
    while { ! [eof $fchannel] } {
	# Stage 1: Source basic variables
	if { $stage == 1 } {
	    if { [string match "set m*" $curr_line] } {
		# Go on to stage 2, not moving on to the next line of the file
		set stage 2
		continue
	    } elseif { [string match "loadfile *" $curr_line] } {
		# do nothing
	    } elseif { [string match "return" $curr_line] } {
		# do nothing
	    } elseif { [string match "puts *" $curr_line] } {
		# do nothing
	    } else {
		# Execute the line (comments and/or blank lines are ignored)
		eval $curr_line
	    }
	}
	
	# Stage 2: Create Modules
	if { $stage == 2 } {
	    if { [string match "set m*" $curr_line] } {
		# build the module
		eval $curr_line
	    } elseif { [string match $curr_line "addConnection*"] } {
		# add connections
		eval $curr_line
	    } elseif { [string match $curr_line ""] } {
		# do nothing
	    } elseif { [string match "addConnection*" $curr_line] } {
		eval $curr_line
	    } else {
		# Move on to the next stage
		set stage 3
		continue
	    }
	}

	# Stage 3: do some stuff
	if { $stage == 3 } {
	    if { [string match "set ::*" $curr_line] } {
		set curr_string $curr_line
		set var [string trimleft $curr_string "set :"]

		set c 0
		set t 0
		set pram ""
		set modname ""
		while { 1 } {
		    set char [string index $var $c]
		    
		    # Check for -'s; if one is found, begin getting
		    # the variable name
		    
		    if { [string match $char "-"] } {
			set t 1
		    }
		    
		    # Break if there is a space
		    
		    if { [string match $char " "] } {
			break
		    }

		    # If the dash has been seen, begin getting the
		    # variable name...

		    if { $t == 1 } {
			set pram "$pram$char"
		    } else {
			set modname "$modname$char"
		    }
		    
		    # increment the counter...
		    incr c
		}
		
		set value [list [lindex $var 1]]
		# Increment the counter each time the modname changes
		
		if { ! [string match $modname $curr_modname] } {
		    incr counter
		    set curr_modname $modname
		}
		
		if { [string match $value ""] } {
		    set value "{}"
		}
		
		set mvar "m$counter"
		set m [expr $$mvar]
		
		set command "set ::$m"
		append command "$pram $value"
		
		# Execute the "real" command
		eval $command
		
		if { [string match "*-group*" $command] } {
		    global $m-group
		    set grp [lindex [set $m-group] 0]
		    if { ! [string match $grp ""] } {
			set group_info "$group_info {$m $grp}"
		    }
		}


	    } elseif { [string match $curr_line ""] } {
		# do nothing
	    } else {
		set stage 4
		continue
	    }
	}
	
	# one last source (this will need to be changed)
	if { $stage == 4 } {
	    eval $curr_line
	}
	
	# Read the next line of the file
	set curr_line [gets $fchannel]
	
	# break out of loop, if at end of file
	if { [eof $fchannel] } {
	    break
	}
    }
    
    # close the file
    close $fchannel


    # set loading back to 0
    set loading 0

    return $group_info
}

proc loadMacroModules {group_info} {
    # Generate group lists
    set mmlist ""
    foreach ginf $group_info {
	set num [lindex $ginf 1]
	set mod [lindex $ginf 0]
	if { [string match "*group$num*" $mmlist] } {
	    set group$num "[set group$num] $mod"
	} else {
	    set group$num $mod
	    set mmlist "$mmlist group$num"
	}
    }


    if { ! [string match $mmlist ""] } {
	# move the modules into their correct positions
	global maincanvas
	global minicanvas

	set mainCanvasWidth 4500
	set mainCanvasHeight 4500

	
	foreach l $mmlist {
	    foreach mod [set $l] {
		global $mod-lastpos
		set lastpos [set $mod-lastpos]
		
		set lastx [lindex $lastpos 0]
		set lasty [lindex $lastpos 1]

		
		set movx [$mod get_x]
		set movy [$mod get_y]
		
		

		set mx [expr -$movx+$lastx]
		set my [expr -$movy+$lasty]
		
		$maincanvas move $mod $mx $my
		

		# account for any scrolling...
		
		set xv [lindex [$maincanvas xview] 0]
		set yv [lindex [$maincanvas yview] 0]

		set xs [expr $xv*$mainCanvasWidth]
		set ys [expr $yv*$mainCanvasWidth]

		$maincanvas move $mod $xs $ys
		

		
		

	    }
	}
	

	foreach l $mmlist {
	    global CurrentlySelectedModules
	    set temp $CurrentlySelectedModules
	    set CurrentlySelectedModules "[set $l]"
	    set curr_mod [lindex $CurrentlySelectedModules 0]
	    set macro [makeMacroModule $maincanvas $minicanvas $curr_mod]
	    rebuildMModuleConnections $macro
	    set CurrentlySelectedModules $temp
	}
    }   
}








