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

#puts "NetworkEditor.tcl start"

source $DataflowTCL/defaults.tcl
source $DataflowTCL/devices.tcl

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

global inserting
set inserting 0

global insertPosition
set insertPosition 0

global modulesBBox
set modulesBbox {0 0 0 0}

global netedit_savefile
set netedit_savefile ""

# List of all currently existing modules
global modules
set modules ""


proc resource {} {
}

proc makeNetworkEditor {} {

    wm minsize . 100 100
    wm geometry . 800x800+0+0

    wm title . "SCIRun"

    frame .main_menu -relief raised -borderwidth 3
    pack .main_menu -fill x
    
    menubutton .main_menu.file -text "File" -underline 0 \
	-menu .main_menu.file.menu
    menu .main_menu.file.menu -tearoff false
    menu .main_menu.file.menu.new -tearoff false
    .main_menu.file.menu.new add command -label "Module..." \
        -underline 0 -command "ComponentWizard"

    # Create the "File" Menu sub-menus.  Create (most of) them in the
    # disabled state.  They will be enabled when all packages are loaded.
    .main_menu.file.menu add command -label "Save" -underline 0 \
	-command "popupSaveMenu" -state disabled
    .main_menu.file.menu add command -label "Save As..." -underline 0 \
	-command "popupSaveAsMenu" -state disabled
    .main_menu.file.menu add command -label "Load..." -underline 0 \
	-command "popupLoadMenu" -state disabled

    .main_menu.file.menu add command -label "Insert" -underline 0 \
	-command "popupInsertMenu" -state disabled
    .main_menu.file.menu add command -label "Clear" -underline 0 \
	-command "ClearCanvas" -state disabled
    .main_menu.file.menu add command -label "Execute All" -underline 0 \
	-command "ExecuteAll" -state disabled
    .main_menu.file.menu add cascade -label "New" -underline 0\
        -menu .main_menu.file.menu.new -state disabled

# This was added by Mohamed Dekhil to add some infor to the net
    .main_menu.file.menu add command -label "Add Info..." -underline 0 \
	-command "popupInfoMenu"

    .main_menu.file.menu add command -label "Quit" -underline 0 \
	    -command "NiceQuit"


    menubutton .main_menu.stats -text "Statistics" -underline 0 \
	-menu .main_menu.stats.menu
    menu .main_menu.stats.menu
    .main_menu.stats.menu add command -label "Memory..." -underline 0 \
	    -command showMemStats
    .main_menu.stats.menu add command -label "Threads..." -underline 0 \
	    -command showThreadStats

#    menubutton .main_menu.help -text "Help" -underline 0 \
#	-menu .main_menu.help.menu
#    menu .main_menu.help.menu
#
#    .main_menu.help.menu add command -label "Help..." -underline 0 \
#	    -command { tk_messageBox -message "I'm helpless" }

#    pack .main_menu.file        \
#         .main_menu.modules     \
#         .main_menu.appModules     -side left
    pack .main_menu.file -side left
#    pack .main_menu.help        \
#         .main_menu.stats          -side right
#    pack .main_menu.stats          -side right

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
    .top.errorFrame.text insert end "--------------------------\n\n"
    .top.errorFrame.text tag configure errtag -foreground red
    .top.errorFrame.text tag configure warntag -foreground orange
    .top.errorFrame.text tag configure infotag -foreground yellow

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
	    [$netedit_canvas yview] 0] + 0.01 ]}
    bind . <KeyPress-Up>    { $netedit_canvas yview moveto [expr [lindex \
	    [$netedit_canvas yview] 0] - 0.01 ] }
    bind . <KeyPress-Left>  { $netedit_canvas xview moveto [expr [lindex \
	    [$netedit_canvas xview] 0] - 0.01 ]} 
    bind . <KeyPress-Right> { $netedit_canvas xview moveto [expr [lindex \
	    [$netedit_canvas xview] 0] + 0.01 ] }
    bind . <Destroy> {if {"%W"=="."} {exit 1}} 
}

proc activate_file_submenus { } {
    # Activate the "File" menu items
    .main_menu.file.menu entryconfig 0 -state active
    .main_menu.file.menu entryconfig 1 -state active
    .main_menu.file.menu entryconfig 2 -state active
    .main_menu.file.menu entryconfig 3 -state active
    .main_menu.file.menu entryconfig 4 -state active
    .main_menu.file.menu entryconfig 5 -state active
    .main_menu.file.menu entryconfig 6 -state active
}

proc handle_bad_startnet { netfile } {
    set answer [tk_messageBox -type ok -parent . -message "Unable to load $netfile as a network.  Exiting." -icon error]
    netedit quit
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
}

proc createPackageMenu {index} {

#  puts "Building Module Menus..."

#  foreach package [netedit packageNames] {
    set packageNames [netedit packageNames]
    set package [lindex $packageNames $index]
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
#  }
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
    global inserting
    
    set mainCanvasWidth 4500
    set mainCanvasHeight 4500
    
    # create the modules at their relative positions only when not loading from a script or inserting.
    
    if { $inserting == 1 } {
	global modulesBbox
	global insertPosition
	if { $insertPosition == 1 } {
	    #inserting net at current screen position
	    set xpos [expr $xpos+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*$mainCanvasWidth))])]
	    set ypos [expr $ypos+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*$mainCanvasHeight))])]
	} else {
	    #inserting net off to the right
	    set xpos [expr int([expr $xpos+[lindex $modulesBbox 2]])]
	     set ypos [expr $ypos+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*$mainCanvasHeight))])]
	}
    } else {
	# place modules as normal
	set xpos [expr $xpos+int([expr (([lindex [.bot.neteditFrame.canvas xview] 0]*$mainCanvasWidth))])]
	set ypos [expr $ypos+int([expr (([lindex [.bot.neteditFrame.canvas yview] 0]*$mainCanvasHeight))])]
    }
    
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
    if { $inserting == 0 } {
	computeModulesBbox
    }
    return $modid
}

proc addModule2 {package category module modid} {
    set className [removeSpaces "${package}_${category}_${module}"]
    if {[catch "$className $modid" exception]} {
	# Use generic module
	if {$exception != "invalid command name \"$className\""} {
	    bgerror "Error instantiating iTcl class for module:\n$exception";
	}
	Module $modid -name "$module"
    }
    return $modid
}

proc addConnection {omodid owhich imodid iwhich} {

    set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
    if {"" == $connid} {
	tk_messageBox -type ok -parent . -message \
	    "Invalid connection found while loading network: addConnection $omodid $owhich $imodid $iwhich -- discarding." \
	    -icon warning
	return
    }
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

    global netedit_savefile
    if { $netedit_savefile != "" } {
	# If we already know the name of the save file, just save it...
	# ...don't ask user for a name.
	saveMacroModules
	netedit savenetwork  $netedit_savefile
    } else {
	# otherwise, get the user involved...
	popupSaveAsMenu
    }
}

proc popupSaveAsMenu {} {
    set types {
	{{SCIRun Net} {.net} }
	{{Uintah Script} {.uin} }
	{{Dataflow Script} {.sr} }
	{{Other} { * } }
    } 
    global netedit_savefile
    set netedit_savefile [ tk_getSaveFile -defaultextension {.net} \
	    -filetypes $types ]
    if { $netedit_savefile != "" } {
	saveMacroModules
	netedit savenetwork  $netedit_savefile

	# Cut off the path from the net name and put in on the title bar:
	set net_name [lrange [split "$netedit_savefile" / ] end end]
	wm title . "SCIRun ($net_name)"
    }
}

proc popupInsertMenu {} {
    global inserting
    set inserting 1
    
    #get the net to be inserted
    set types {
	{{SCIRun Net} {.net} }
	{{Uintah Script} {.uin} }
	{{Dataflow Script} {.sr} }
	{{Other} { * } }
    } 
    set netedit_loadnet [tk_getOpenFile -filetypes $types ]
    
    if { [file exists $netedit_loadnet] } {
	# get the bbox for the net being inserted by
	# parsing netedit_loadnet for bbox 
	global modulesBbox
	set insertBbox {0 0 4500 4500}
	set fchannel [open $netedit_loadnet]
	set curr_line ""
	set curr_line [gets $fchannel]
	while { ![eof $fchannel] } {
	    if { [string match "set bbox*" $curr_line] } {
		set bbox {0 0 4500 4500}
		eval $curr_line
		set insertBbox $bbox
		break
	    }
	    set curr_line [gets $fchannel]
	}
	#determine if the inserted net should go on the 
	#current canvas view or to the right of the bbox
	global mainCanvasWidth mainCanvasHeight
	global insertPosition
	set x [expr [lindex [.bot.neteditFrame.canvas xview] 0]*\
		$mainCanvasWidth]
	set y [expr [lindex [.bot.neteditFrame.canvas yview] 0]*\
		$mainCanvasHeight] 
	if { $x > [lindex $modulesBbox 2] || $y > [lindex $modulesBbox 3] } {
	    # far enough down or to the right to fit on current position
	    set insertPosition 1
	} else {
	    set width [expr [lindex $insertBbox 2]-[lindex $insertBbox 0]]
	    set height [expr [lindex $insertBbox 3]-[lindex $insertBbox 1]]
	    set startX [expr $x+[lindex $insertBbox 0]]
	    set endX [expr $startX+$width]
	    set startY [expr $y+[lindex $insertBbox 1]]
	    set endY [expr $startY+$height]
	    if { [expr $startX+$width] < [lindex $modulesBbox 0] || \
		    [expr $startY+$height] < [lindex $modulesBbox 1] } {
		# net to be inserted will fit at current position
		# and not be in bbox
		set insertPosition 1
	    } else {
		global maincanvas
		global modules
		if { [info exists modules] == 1} {
		    set fits 1
		    foreach m $modules {
			set curr_coords [$maincanvas coords $m]
			if { [lindex $curr_coords 0] < $endX && \
				[lindex $curr_coords 1] > $startX } {
			    if { [lindex $curr_coords 1] < $endY && \
				    [lindex $curr_coords 1] > $startY } {
				set fits 0
				break
			    }
			}
		    }
		    if { $fits == 1 } {
			# enough room within the modulesBbox to
			# fit the net
			set insertPosition 1
		    } else {
			# insert net to the right
			set insertPosition 0
		    }
		} else {
		    # insert was first action so put net
		    # at current position
		    set insertPosition 1
		}
	    }
	}
    	loadnet $netedit_loadnet
    }
    set inserting 0
    computeModulesBbox
}

proc computeModulesBbox {} {
    global maincanvas
    global modules
    set maxx 0
    set maxy 0
    
    set minx 4500
    set miny 4500
    
    global modules
    if { $modules == "" } {
	set maxx 0
	set maxy 0
	set minx 0
	set miny 0
    } 
    foreach m $modules {
	set curr_coords [$maincanvas coords $m]
	
	#Find $maxx and $maxy
	if { [lindex [$maincanvas bbox $m] 2] > $maxx} {
	    set maxx [lindex [$maincanvas bbox $m] 2]
	}
	if { [lindex [$maincanvas bbox $m] 3] > $maxy} {
	    set maxy [lindex [$maincanvas bbox $m] 3]
	}
	
	#Find $minx and $miny
	
	if { [lindex $curr_coords 0] <= $minx} {
	    set minx [lindex $curr_coords 0]
	}
	if { [lindex $curr_coords 1] <= $miny} {
	    set miny [lindex $curr_coords 1]
	}
    }
    
    global modulesBbox
    set modulesBbox "$minx $miny $maxx $maxy"
}

proc popupLoadMenu {} {
    global netedit
    set types {
	{{SCIRun Net} {.net} }
	{{Uintah Script} {.uin} }
	{{Dataflow Script} {.sr} }
	{{Other} { * } }
    } 
    
    set netedit_loadnet [tk_getOpenFile -filetypes $types ]
    
    if { [file exists $netedit_loadnet] } {
	loadnet $netedit_loadnet
    }
}

proc ClearCanvas {} {
   # destroy all modules
    
    set result [tk_messageBox -type okcancel -parent . -message \
	        "ALL modules and connections will be cleared.\nReally clear?"\
		-icon warning ]
    
    if {[string compare "ok" $result] == 0} {
	global modules
	if { [info exists modules] } {
	    foreach m $modules {
		moduleDestroy .bot.neteditFrame.canvas \
			.top.globalViewFrame.canvas $m
	    }
	}    

	# Reset title of main window:
	wm title . "SCIRun"

	# reset all the NetworkEditor globals to their initial values
	set mainCanvasWidth    4500.0
	set mainCanvasHeight   4500.0
	set miniCanvasWidth     150.0
	set miniCanvasHeight    150.0
	set SCALEX [expr $mainCanvasWidth/$miniCanvasWidth]
	set SCALEY [expr $mainCanvasHeight/$miniCanvasHeight]
	
	set mouseX 0
	set mouseY 0
	
	global maincanvas
	set maincanvas ".bot.neteditFrame.canvas"
	global minicanvs
	set minicanvas ".top.globalViewFrame.canvas"
	
	global loading
	set loading 0
	
	global inserting
	set inserting 0
	
	global insertPosition
	set insertPosition 0
	
	global modulesBBox
	set modulesBbox {0 0 0 0}
	
	global netedit_savefile
	set netedit_savefile ""
	
	#reset Module.tcl variables
	global connection_list
	set connection_list ""
	
	global selected_color
	set selected_color darkgray
	
	global unselected_color
	set unselected_color gray
	
	global MModuleFakeConnections
	set MModuleFakeConnections ""
	
	global CurrentlySelectedModules
	set CurrentlySelectedModules ""
	
	global CurrentMacroModules
	set CurrentMacroModules ""
	
	global MacroedModules
	set MacroedModules ""
	
	global modules
	set modules ""
    }   
}

proc NiceQuit {} {

    set result [tk_messageBox -type okcancel -parent . -message \
		    "Please confirm exit." -icon warning ]

    if {[string compare "ok" $result] == 0} {
	# Disable this for now, module deletion is blocking.
	# This causes quit to get stuck when the modules are running.
        #
	#global modules
	#if { [info exists modules] } {
	#    foreach m $modules {
	#	moduleDestroy .bot.neteditFrame.canvas .top.globalViewFrame.canvas $m
	#    }
	#}    
	set modules ""
	puts "Goodbye!"
	netedit quit
    }   
}


proc ExecuteAll {} {
    netedit scheduleall
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
    puts "NOTICE: `loadfile' has been disabled."
    puts "   To use old nets, remove the `loadfile' and `return' lines"
    puts "   from near the top of the file."
    return
}

proc loadnet {netedit_loadfile} {
    global loading
    set loading 0
    set group_info [sourcenet $netedit_loadfile]
    set loading 1

    if { ! [string match $group_info ""] } {
	[loadMacroModules $group_info]
    }
}

proc sourcenet {netedit_loadfile} {
    # Check to see of the file exists; exit if it doesn't
    if { ! [file exists $netedit_loadfile] } {
	handle_bad_startnet "$netedit_loadfile"
	return
    }

    # Cut off the path from the net name and put in on the title bar:
    set net_name [lrange [split "$netedit_loadfile" / ] end end]
    wm title . "SCIRun ($net_name)"

    # Remember the name of this net for future "Saves".
    global netedit_savefile
    set netedit_savefile $netedit_loadfile

    # I believe that we use this 'global' because the "file_to_load"
    # variable is not seen in the "source" call below if it is a 
    # local variable... I'm not sure why.
    global file_to_load
    set file_to_load $netedit_loadfile

    # The '#' below is not a comment...
    uplevel #0 {source $file_to_load}
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

#    set stage 1
# DMW: without macromodules, we can just source all of the lines of the file
#          which is what stage 4 does.

    set stage 4

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


# DMW: for backwards compatability, we need to ignore the loadfile and return
#          lines

	if { $stage == 4 } {
	    if { ![string match "load*" $curr_line] } {
		if { ![string match "return*" $curr_line] } {
		    puts $curr_line
		    eval $curr_line
		}
	    }
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

#
# Ask the user to select a data directory (because the enviroment variable 
# SCIRUN_DATA was not set.)
#
proc getDataDirectory { dataset } {
    tk_messageBox -type ok -parent . -message \
         "The '$dataset' dataset was specified (either by the enviroment variable SCIRUN_DATASET or by the network loaded).  However, the location of this dataset was not specified (with the SCIRUN_DATA env var).  Please select a directory (eg: /usr/sci/data/SCIRunData/1.10.0).  Note, this directory must have the '$dataset' subdirectory in it." 
   return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.10.0]
}

#
# Tell the user the reason that they are being asked for the data, and
# then ask for the data.  If "warn_user" is "true", warning is displayed,
# otherwise we bring up the choose directory dialog directly.  "warn_user"
# most likely should only be displayed the first time.
#
proc getSettingsDirectory { warn_user } {

   if { "$warn_user" == "true" } {
      tk_messageBox -type ok -parent . -message \
         "The enviroment variables SCIRUN_DATA and/or SCIRUN_DATASET are not set (or are invalid).  You must specify a valid data set directory in order to use this net!  You will now be asked to select the directory of the dataset you are interested in.  (eg: /usr/sci/data/SCIRunData/1.10.0/sphere) (FYI, if you set these environment variables, you will not need to select a directory manually when you load this network.)" 
   }
   return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.10.0]
}

#
# Verify that the "file_name" file exists.  If not, tells the user that
# they will next have to input valid data.  It is up to the calling routine
# to check if verifyFile() returns "true", and if not, to ask the user for
# new information.
#
proc verifyFile { file_name } {
  if {![file isfile $file_name]} {

     set message "Error: $file_name is not a valid file.  Please select a valid directory (eg: /usr/sci/data/SCIRunData/1.10.0/sphere)"

     # This occurs if the user presses "cancel".
     if { "$file_name" == "//.settings" } {
        set message "You must select a data set directory for use with this network. Eg: /usr/sci/data/SCIRunData/1.10.0/sphere"
     }

     tk_messageBox -type ok -parent . -message "$message" -icon warning

     return "false"
  }
  return "true"
}

#
# sourceSettingsFile()
#
# Finds and then sources the ".settings" file.  Uses environment
# variables SCIRUN_DATA (for directory) and SCIRUN_DATASET (for data
# set) to find .settings file.  If these environment variables are not
# set or if they to not point to a valid file, then proc asks the user
# for input.  
#
# Returns "DATADIR DATASET"
#
proc sourceSettingsFile {} {

   global env

   # Attempt to get environment variables:
   set DATADIR [lindex [array get env SCIRUN_DATA] 1]
   set DATASET [lindex [array get env SCIRUN_DATASET] 1]

   if { [string compare "$DATASET" ""] == 0 } {
       # if env var SCIRUN_DATASET not set... default to sphere:
       set DATASET "sphere"
   } else {
       if { [string compare "$DATADIR" ""] == 0 } {
          # DATASET specified, but not DATADIR.  Ask for DATADIR
          set DATADIR [getDataDirectory $DATASET]
       }
   }

   if { [string compare "$DATADIR" ""] != 0 && \
        [verifyFile $DATADIR/$DATASET/$DATASET.settings] == "true" } {
      displayErrorWarningOrInfo "*** Using SCIRUN_DATA $DATADIR" "info"
      displayErrorWarningOrInfo "*** Using DATASET $DATASET" "info"
   } else {
      set done "false"
      set warn_user "true"
      while { $done == "false" } {

         # "result" is the directory of the dataset (eg: /usr/sci/data/sphere)
	 set result [getSettingsDirectory "$warn_user"]
         # cut off beginning: if /my/data/sets/sphere, this gives "sphere"
	 set DATASET [lrange [split "$result" / ] end end]
         # cut off end: if /my/data/sets/sphere, this gives "/my/data/sets"
	 set DATADIR [string range "$result" 0 \
		[expr [string length $result] - [string length $DATASET] - 2]]

         if { [verifyFile $DATADIR/$DATASET/$DATASET.settings] == "true" } {
            displayErrorWarningOrInfo "*** Using SCIRUN_DATA $DATADIR" "info"
            displayErrorWarningOrInfo "*** Using DATASET $DATASET" "info"
            set done "true"
	 }
         set warn_user "false"
      }
   }
   source $DATADIR/$DATASET/$DATASET.settings
   return "$DATADIR $DATASET"
}

#
# displayErrorWarningOrInfo(): 
#
# Generic function that should be called to display to Error/Warning
# window.  (Ie: don't write to the window directly, use this function.)
#
# Displays the message "msg" (a string) to the Error/Warning window.
# "status" (Possible values: "error", "warning", "info") determines if
# the message should be displayed as an error, as a warning, or as
# information.
#
proc displayErrorWarningOrInfo { msg status } {

    # Yellow Message
    set status_tag "infotag"
    if { "$status" == "error" } {
       # Red Message
       set status_tag "errtag" 
    } elseif { "$status" == "warning" } {
       # Orange Message
       set status_tag "warntag" 
    }
    .top.errorFrame.text insert end "$msg\n" "$status_tag"
    .top.errorFrame.text see end
}

