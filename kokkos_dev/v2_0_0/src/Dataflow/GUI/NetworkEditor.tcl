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
global mouseX
set mouseX 0
global mouseY
set mouseY 0

global maincanvas
set maincanvas ".bot.neteditFrame.canvas"
global minicanvas
set minicanvas ".top.globalViewFrame.canvas"
global Subnet
set Subnet(Subnet0_minicanvas) $minicanvas
set Subnet(num) 0
set Subnet(Subnet0_canvas) $maincanvas
set Subnet(Subnet0_Name) "Main"
set Subnet(Subnet0_Modules) ""
set Subnet(Subnet0_connections) ""
set Subnet(Subnet0) 0


global inserting
set inserting 0

global netedit_savefile
set netedit_savefile ""

global NetworkChanged
set NetworkChanged 0

# Make sure version stays in sync with main/main.cc
global SCIRun_version
set SCIRun_version v1.20.3


proc makeNetworkEditor {} {

    wm protocol . WM_DELETE_WINDOW { NiceQuit }
    wm minsize . 100 100
    wm geometry . 800x800+0+0
    wm title . "SCIRun"

    loadToolTipText

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
    .main_menu.file.menu add command -label "Load..." -underline 0 \
	-command "popupLoadMenu" -state disabled
    .main_menu.file.menu add command -label "Insert..." -underline 0 \
	-command "popupInsertMenu" -state disabled
    .main_menu.file.menu add command -label "Save" -underline 0 \
	-command "popupSaveMenu" -state disabled
    .main_menu.file.menu add command -label "Save As..." -underline 0 \
	-command "popupSaveAsMenu" -state disabled

    .main_menu.file.menu add separator

    .main_menu.file.menu add command -label "Clear Network" -underline 0 \
	-command "ClearCanvas" -state disabled
    .main_menu.file.menu add command -label "Select All" -underline 0 \
	-command "selectAll" -state disabled

    if 0 {
        .main_menu.file.menu add separator
	.main_menu.file.menu add command -label "Save Postscript..." -underline 0 \
	    -command ".bot.neteditFrame.canvas postscript -file /tmp/canvas.ps -x 0 -y 0 -width 4500 -height 4500" -state disabled
    }
    .main_menu.file.menu add command -label "Execute All" -underline 0 \
	-command "netedit scheduleall" -state disabled

    .main_menu.file.menu add separator
    .main_menu.file.menu add cascade -label "New" -underline 0\
        -menu .main_menu.file.menu.new -state disabled
    .main_menu.file.menu add separator

    # This was added by Mohamed Dekhil to add some infor to the net
    .main_menu.file.menu add command -label "Add Info..." -underline 0 \
	-command "popupInfoMenu"

    .main_menu.file.menu add separator
    .main_menu.file.menu add command -label "Quit" -underline 0 \
	    -command "NiceQuit"


    pack .main_menu.file -side left
    global ToolTipText
    Tooltip .main_menu.file $ToolTipText(FileMenu)
    
    menubutton .main_menu.subnet -text "Sub-Networks" -underline 0 \
	-menu .main_menu.subnet.menu -direction below
    menu .main_menu.subnet.menu -tearoff false
    pack .main_menu.subnet -side left
    .main_menu.subnet configure -state disabled


    menubutton .main_menu.help -text "Help" -underline 0 \
	-menu .main_menu.help.menu -direction below
    menu .main_menu.help.menu -tearoff false
    .main_menu.help.menu add checkbutton -label "Show Tooltips" -underline 0 \
	-variable tooltipsOn

    # Mac hack to fix size of 'About' window ... sigh... 
    .main_menu.help.menu add command -label "About..." -underline 0 -state disabled \
	-command  "showSplash main/scisplash.ppm; after 0 {wm geometry .splash \"\"}"

    .main_menu.help.menu add command -label "License..." -underline 0 \
	-command  "licenseDialog" -state disabled

    pack .main_menu.help -side right
    Tooltip .main_menu.help $ToolTipText(HelpMenu)
    
    tk_menuBar .main_menu .main_menu.file

    frame .top -borderwidth 5
    pack  .top -side top -fill x
    frame .bot -borderwidth 5
    pack  .bot -side bottom -expand yes -fill both

    frame .top.globalViewFrame -relief sunken -borderwidth 3
    frame .bot.neteditFrame -relief sunken -borderwidth 3

    global maincanvas minicanvas mainCanvasHeight mainCanvasWidth Color
    canvas $maincanvas -bg "$Color(NetworkEditor)" \
        -scrollregion "0 0 $mainCanvasWidth $mainCanvasHeight" \
	
    # bgRect is just a rectangle drawn on the neteditFrame Canvas
    # so that the Modules List Menu can be bound to it using mouse
    # button 3.  The Modules List Menu can't be bound to the canvas
    # itself because mouse events are sent to both the objects on the
    # canvas (such as the lines connection the modules) and the canvas.

    eval $maincanvas create rectangle [$maincanvas cget -scrollregion] \
	-fill "$Color(NetworkEditor)" -tags bgRect



    scrollbar .bot.neteditFrame.hscroll -relief sunken -orient horizontal \
	-command "$maincanvas xview"
    scrollbar .bot.neteditFrame.vscroll -relief sunken \
	-command "$maincanvas yview" 

    pack .bot.neteditFrame -expand yes -fill both -padx 4

    
    grid $maincanvas .bot.neteditFrame.vscroll .bot.neteditFrame.hscroll
    grid columnconfigure .bot.neteditFrame 0 -weight 1 
    grid rowconfigure    .bot.neteditFrame 0 -weight 1 

    grid config $maincanvas -column 0 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "snew" 
    grid config .bot.neteditFrame.hscroll -column 0 -row 1 \
	    -columnspan 1 -rowspan 1 -sticky "ew" -pady 2
    grid config .bot.neteditFrame.vscroll -column 1 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "sn" -padx 2
    # Create Error Message Window...
    frame .top.errorFrame -borderwidth 3 
    text .top.errorFrame.text -relief sunken -bd 3 \
	-bg "$Color(ErrorFrameBG)" -fg "$Color(ErrorFrameFG)" \
	-yscrollcommand ".top.errorFrame.s set" -height 10 -width 180 
    .top.errorFrame.text insert end "Messages:\n"
    .top.errorFrame.text insert end "--------------------------\n\n"
    .top.errorFrame.text tag configure errtag -foreground red
    .top.errorFrame.text tag configure warntag -foreground orange
    .top.errorFrame.text tag configure infotag -foreground yellow


    scrollbar .top.errorFrame.s -relief sunken \
	    -command ".top.errorFrame.text yview"
    pack .top.errorFrame.s -side right -fill y -padx 4
    pack .top.errorFrame.text -expand yes -fill both
    global netedit_errortext
    set netedit_errortext .top.errorFrame.text

    pack .top.globalViewFrame -side left -padx 4
    pack .top.errorFrame -side right -fill both -expand yes

    global miniCanvasWidth miniCanvasHeight
    canvas $minicanvas -bg $Color(NetworkEditor) \
	-width $miniCanvasWidth -height $miniCanvasHeight
    pack $minicanvas 

    $minicanvas create rectangle 0 0 1 1 -outline black -tag "viewAreaBox"
    $maincanvas configure \
	-xscrollcommand "updateCanvasX" -yscrollcommand "updateCanvasY"

    wm withdraw .
}

proc canvasScroll { canvas { dx 0.0 } { dy 0.0 } } {
    if {$dx!=0.0} {$canvas xview moveto [expr $dx+[lindex [$canvas xview] 0]]}
    if {$dy!=0.0} {$canvas yview moveto [expr $dy+[lindex [$canvas yview] 0]]}
}

# Activate the "File" menu items - called from C after all packages are loaded
proc activate_file_submenus { } {
    .main_menu.file.menu entryconfig  0 -state active
    .main_menu.file.menu entryconfig  1 -state active
    .main_menu.file.menu entryconfig  2 -state active
    .main_menu.file.menu entryconfig  3 -state active
    .main_menu.file.menu entryconfig  5 -state active
    .main_menu.file.menu entryconfig  6 -state active
    .main_menu.file.menu entryconfig  7 -state active
    .main_menu.file.menu entryconfig  9 -state active
    .main_menu.file.menu entryconfig 11 -state active

    .main_menu.help.menu entryconfig  1 -state active
    .main_menu.help.menu entryconfig  2 -state active

    ###################################################################
    # Bind all the actions after SCIRun has loaded everything...
    global maincanvas minicanvas

    bind $minicanvas <B1-Motion> "updateCanvases %x %y"
    bind $minicanvas <1> "updateCanvases %x %y"
    bind $maincanvas <Configure> "handleResize %w %h"
    $maincanvas bind bgRect <3> "modulesMenu 0 %x %y"
    $maincanvas bind bgRect <1> "startBox $maincanvas %X %Y 0"
    $maincanvas bind bgRect <Control-Button-1> "startBox $maincanvas %X %Y 1"
    $maincanvas bind bgRect <B1-Motion> "makeBox $maincanvas %X %Y"
    $maincanvas bind bgRect <ButtonRelease-1> "$maincanvas delete tempbox"

    # Canvas up-down bound to mouse scroll wheel
    bind . <ButtonPress-5>  "canvasScroll $maincanvas 0.0 0.01"
    bind . <ButtonPress-4>  "canvasScroll $maincanvas 0.0 -0.01"
    # Canvas movement on arrow keys press
    bind . <KeyPress-Down>  "canvasScroll $maincanvas 0.0 0.01"
    bind . <KeyPress-Up>    "canvasScroll $maincanvas 0.0 -0.01"
    bind . <KeyPress-Left>  "canvasScroll $maincanvas -0.01 0.0"
    bind . <KeyPress-Right> "canvasScroll $maincanvas 0.01 0.0" 
    bind . <Destroy> {if {"%W"=="."} {exit 1}} 

    # Destroy selected items with a Ctrl-D press
    bind all <Control-d> "moduleDestroySelected"
    # Clear the canvas
    bind all <Control-l> "ClearCanvas"
    bind all <Control-z> "undo"
    bind all <Control-a> "selectAll"
    bind all <Control-y> "redo"
}

proc modulesMenu { subnet x y } {
    global mouseX mouseY Subnet
    set mouseX $x
    set mouseY $y
    set canvas $Subnet(Subnet${subnet}_canvas)
    createModulesMenu $subnet
    tk_popup $canvas.modulesMenu [expr $x + [winfo rootx $canvas]] \
	[expr $y + [winfo rooty $canvas]]
}

proc handleResize { w h } {
    global minicanvas maincanvas SCALEX SCALEY
    set ulx [lindex [$minicanvas coords viewAreaBox] 0]
    set uly [lindex [$minicanvas coords viewAreaBox] 1]
    set lrx [expr $ulx + [winfo width  $maincanvas]/$SCALEX]
    set lry [expr $uly + [winfo height $maincanvas]/$SCALEY ]
    $minicanvas coords viewAreaBox $ulx $uly $lrx $lry
}

proc updateCanvasX { beg end } {
    global maincanvas minicanvas SCALEX SCALEY miniCanvasWidth miniCanvasHeight
    # Tell the scroll bar to update
    .bot.neteditFrame.hscroll set $beg $end
    # Update the view area box 
    set uly [lindex [$minicanvas coords viewAreaBox] 1]
    set lry [lindex [$minicanvas coords viewAreaBox] 3]
    set ulx [expr $beg * $miniCanvasWidth ]
    set lrx [expr $ulx + [winfo width $maincanvas]/$SCALEX - 1 ]
    $minicanvas coords viewAreaBox $ulx $uly $lrx $lry
}

proc updateCanvasY { beg end } {
    global maincanvas minicanvas SCALEX SCALEY miniCanvasWidth miniCanvasHeight
    # Tell the scroll bar to update
    .bot.neteditFrame.vscroll set $beg $end
    # Update the view area box 
    set ulx [lindex [$minicanvas coords viewAreaBox] 0]
    set uly [expr $beg * $miniCanvasHeight ]
    set lrx [lindex [$minicanvas coords viewAreaBox] 2]
    set lry [expr $uly + [winfo height $maincanvas]/$SCALEY - 1 ]
    $minicanvas coords viewAreaBox $ulx $uly $lrx $lry
}

proc updateCanvases { x y } {
    global miniCanvasWidth miniCanvasHeight maincanvas minicanvas
    # Find the width and height of the mini box.
    set boxBbox [$minicanvas coords viewAreaBox]
    # Store 1/2 Width and 1/2 Height of the mini box
    set wid [expr ([lindex $boxBbox 2] - [lindex $boxBbox 0])/2]
    set hei [expr ([lindex $boxBbox 3] - [lindex $boxBbox 1])/2]
    if { $x < $wid } { set x $wid }
    if { $x > ($miniCanvasWidth - $wid) } \
	{ set x [expr $miniCanvasWidth - $wid - 1] }
    if { $y < $hei } { set y $hei }
    if { $y > ($miniCanvasHeight - $hei) } \
         { set y [expr $miniCanvasHeight - $hei - 1] }
    # Move the minibox to the new location
    $minicanvas coords viewAreaBox \
	[expr $x - $wid] [expr $y - $hei] [expr $x + $wid] [expr $y + $hei]
    # Update the region displayed in the main canvas.
    $maincanvas xview moveto [expr ($x - $wid)/$miniCanvasWidth]
    $maincanvas yview moveto [expr ($y - $hei)/$miniCanvasHeight]
}

proc createPackageMenu {index} {
    global ModuleMenu
    set package [lindex [netedit packageNames] $index]
    set packageToken [join "menu_${package}" ""]
    set ModuleMenu($packageToken) $package
    lappend ModuleMenu(packages) $packageToken
    foreach category [netedit categoryNames $package] {	
	set categoryToken [join "${packageToken}_${category}" ""]
	set ModuleMenu($categoryToken) $category
	lappend ModuleMenu(${packageToken}_categories) $categoryToken
	foreach module [netedit moduleNames $package $category] {
	    set moduleToken [join "${categoryToken}_${module}" ""]
	    set ModuleMenu($moduleToken) $module
	    lappend ModuleMenu(${packageToken}_${categoryToken}_modules) $moduleToken
	}
    }

    set pack $packageToken
    # Add the cascade button and menu for the package to the menu bar
    menubutton .main_menu.$pack -text "$ModuleMenu($pack)" -underline 0 \
	-menu .main_menu.$pack.menu
    menu .main_menu.$pack.menu
    pack forget .main_menu.subnet
    pack .main_menu.$pack -side left
    pack .main_menu.subnet -side left
    global ToolTipText
    Tooltip .main_menu.$pack $ToolTipText(PackageMenus)

    foreach cat $ModuleMenu(${pack}_categories) {
	# Add the category to the menu bar menu
	.main_menu.$pack.menu add cascade -label "$ModuleMenu($cat)" \
	    -menu .main_menu.$pack.menu.$cat
	menu .main_menu.$pack.menu.$cat -tearoff false
	foreach mod $ModuleMenu(${pack}_${cat}_modules) {
	    .main_menu.$pack.menu.$cat add command \
		-label "$ModuleMenu($mod)" \
		-command "addModule \"$ModuleMenu($pack)\" \"$ModuleMenu($cat)\" \"$ModuleMenu($mod)\""
	}
    }

    createModulesMenu 0
    update idletasks
}

proc createModulesMenu { subnet } {
    global ModuleMenu Subnet
    if ![info exists ModuleMenu] return
    set canvas $Subnet(Subnet${subnet}_canvas)
    if [winfo exists $canvas.modulesMenu] {	
	destroy $canvas.modulesMenu
    }
	
    if ![winfo exists $canvas.modulesMenu] {	
	menu $canvas.modulesMenu -tearoff false
	foreach pack $ModuleMenu(packages) {
	    # Add a separator to the right-button menu for this package if this
	    # isn't the first package to go in there
	    if { [$canvas.modulesMenu index end] != "none" } \
		{ $canvas.modulesMenu add separator }
	    $canvas.modulesMenu add command -label "$ModuleMenu($pack)"
	    foreach cat $ModuleMenu(${pack}_categories) {
		# Add the category to the right-button menu
		$canvas.modulesMenu add cascade -label "  $ModuleMenu($cat)" \
		    -menu $canvas.modulesMenu.$cat
		menu $canvas.modulesMenu.$cat -tearoff false
		foreach mod $ModuleMenu(${pack}_${cat}_modules) {
		    $canvas.modulesMenu.$cat add command \
			-label "$ModuleMenu($mod)" \
			-command "global Subnet
                              set Subnet(Loading) $subnet
                              addModuleAtMouse \"$ModuleMenu($pack)\" \"$ModuleMenu($cat)\" \"$ModuleMenu($mod)\"
                              set Subnet(Loading) 0"
		}
	    }
	}
	$canvas.modulesMenu add separator
	$canvas.modulesMenu add cascade -label "Sub-Networks" \
	    -menu $canvas.modulesMenu.subnet
	menu $canvas.modulesMenu.subnet -tearoff false
    }

    createSubnetMenu $subnet
	
    update idletasks
}

proc createSubnetMenu { { subnet 0 } } {
    global SCIRUN_SRCDIR Subnet 
    set filelist1 [glob -nocomplain -dir $SCIRUN_SRCDIR/Subnets *.net]
    set filelist2 [glob -nocomplain -dir ~/SCIRun/Subnets *.net]
    set subnetfiles [lsort -dictionary [concat $filelist1 $filelist2]]    
    set menu $Subnet(Subnet${subnet}_canvas).modulesMenu
    
    $menu.subnet delete 0 end
    .main_menu.subnet.menu delete 0 end

    if ![llength $subnetfiles] {
	$menu entryconfigure Sub-Networks -state disabled
	.main_menu.subnet configure -state disabled
    } else {
	$menu entryconfigure Sub-Networks -state normal
	.main_menu.subnet configure -state normal
	foreach file $subnetfiles {
	    set filename [lindex [file split $file] end]
	    set name [join [lrange [split  $filename "."] 0 end-1] "."]
	    $menu.subnet add command -label "$name" \
		-command "global Subnet
			      set Subnet(Loading) $subnet
			      loadSubnet {$file}
                              set Subnet(Loading) 0"
	    
	    .main_menu.subnet.menu add command -label "$name" \
		-command "global Subnet
                          set Subnet(Loading) 0
                          loadSubnet {$file}"
	}
    }
}
    

proc networkHasChanged {args} {
    global NetworkChanged
    set NetworkChanged 1
}

proc addModule { package category module } {
    return [addModuleAtPosition "$package" "$category" "$module" 10 10]
}

proc addModuleAtMouse { package category module } {
    global mouseX mouseY
    return [ addModuleAtPosition "$package" "$category" "$module" $mouseX \
             $mouseY ]
}

proc addModuleAtPosition {package category module { xpos 10 } { ypos 10 } } {
    # Look up the real category for a module.  This allows networks to
    # be read in if the modules change categories.
    set category [netedit getCategoryName $package $category $module]
    set modid [netedit addmodule "$package" "$category" "$module"]

    # netedit addmodule returns an empty string if the module wasnt created
    if { ![string length $modid] } {
	tk_messageBox -type ok -parent . -icon warning -message \
	    "Cannot find the ${package}::${category}::${module} module."
	return
    }    

    networkHasChanged
    global inserting Subnet
    set canvas $Subnet(Subnet$Subnet(Loading)_canvas)
    set Subnet($modid) $Subnet(Loading)
    set Subnet(${modid}_connections) ""
    lappend Subnet(Subnet$Subnet(Loading)_Modules) $modid

    set className [join "${package}_${category}_${module}" ""]
    # Create the itcl object
    if {[catch "$className $modid" exception]} {
	# Use generic module
	if {$exception != "invalid command name \"$className\""} {
	    bgerror "Error instantiating iTcl class for module:\n$exception";
	}
	Module $modid -name "$module"
    }

    # compute position if we're inserting the net to the right    
    if { $inserting } {
	global insertOffset
	set xpos [expr $xpos+[lindex $insertOffset 0]]
	set ypos [expr $ypos+[lindex $insertOffset 1]]
    } else { ;# create the module relative to current screen position
	set xpos [expr $xpos+[$canvas canvasx 0]]
	set ypos [expr $ypos+[$canvas canvasy 0]]
    }
    $modid make_icon $xpos $ypos $inserting
    update idletasks
    return $modid
}

proc addModule2 {package category module modid} {
    set className [join "${package}_${category}_${module}" ""]
    if {[catch "$className $modid" exception]} {
	# Use generic module
	if {$exception != "invalid command name \"$className\""} {
	    bgerror "Error instantiating iTcl class for module:\n$exception";
	}
	Module $modid -name "$module"
    }
    return $modid
}


proc popupSaveMenu {} {
    global netedit_savefile NetworkChanged
    if { $netedit_savefile != "" } {
	# We know the name of the savefile, dont ask for name, just save it
	netedit savenetwork $netedit_savefile 0
	set NetworkChanged 0
    } else { ;# Otherwise, ask the user for the name to save as
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
    global netedit_savefile NetworkChanged
    set netedit_savefile \
	[tk_getSaveFile -defaultextension {.net} -filetypes $types ]
    if { $netedit_savefile != "" } {
	netedit savenetwork $netedit_savefile 0
	set NetworkChanged 0
	# Cut off the path from the net name and put in on the title bar:
	wm title . "SCIRun ([lindex [split "$netedit_savefile" /] end])"
    }
}

proc popupInsertMenu { {subnet 0} } {
    global inserting insertOffset Subnet NetworkChanged
    global mainCanvasWidth mainCanvasHeight
    
    #get the net to be inserted
    set types {
	{{SCIRun Net} {.net} }
	{{Uintah Script} {.uin} }
	{{Dataflow Script} {.sr} }
	{{Other} { * } }
    } 
    set netedit_loadnet [tk_getOpenFile -filetypes $types ]
    if { $netedit_loadnet == "" || ![file exists $netedit_loadnet]} { 
	return
    }
    
    set canvas $Subnet(Subnet${subnet}_canvas)    
    # get the bbox for the net being inserted by
    # parsing netedit_loadnet for bbox 
    set fchannel [open $netedit_loadnet]
    set curr_line ""
    set curr_line [gets $fchannel]
    while { ![eof $fchannel] } {
	if { [string match "set bbox*" $curr_line] } {
	    eval $curr_line
	    break
	}
	set curr_line [gets $fchannel]
	set val [string match "set bbox*" $curr_line]

    }
    set viewBox "0 0 [winfo width $canvas] [winfo width $canvas]"
    if { ![info exists bbox] || [llength $bbox] != 4 } {
	set bbox $viewBox
    }
    set w [expr [lindex $bbox 2] - [lindex $bbox 0]]
    set h [expr [lindex $bbox 3] - [lindex $bbox 1]]
    set oldbox $bbox
    set moveBoxX "[expr $w/2] 0 [expr $w/2] 0"
    set moveBoxY "0 [expr $h/2] 0 [expr $h/2]"
    set done 0
    set bbox [list 0 0 $w $h] ;# start inserting in upper left corner
    while {!$done} {
	set done 1
	set modules [eval $canvas find overlapping $bbox]
	foreach modid $modules {
	    if { [lsearch [$canvas gettags $modid] module] != -1 } {
		set overlap [clipBBoxes [compute_bbox $canvas $modid] $bbox]
		if ![string equal $overlap "0 0 0 0"] {
		    set done 0
		    break
		}
	    }
	}
	if {!$done} {
	    # move the insert position left by half a screen
	    for {set i 0} {$i < 4} {incr i} {
		set bbox [lreplace $bbox $i $i \
			      [expr [lindex $moveBoxX $i]+[lindex $bbox $i]]]
	    }
	    if {[lindex $bbox 2] > $mainCanvasWidth} {
		set bbox [lreplace $bbox 2 2 \
			  [expr [lindex $bbox 2] -[lindex $bbox 0]]]
		set bbox [lreplace $bbox 0 0 0]
		for {set i 0} {$i < 4} {incr i} {
		    set bbox [lreplace $bbox $i $i \
			       [expr [lindex $moveBoxY $i]+[lindex $bbox $i]]]
		}
		if {[lindex $bbox 3] > $mainCanvasHeight} {
		    set bbox [list 50 50 [expr 50+$w] [expr 50+$h]]
		    set done 1
		}
	    }
	}
    }

    set insertOffset [list [expr [lindex $bbox 0]-[lindex $oldbox 0]] \
			  [expr [lindex $bbox 1]-[lindex $oldbox 1]]]
    $canvas xview moveto [expr [lindex $bbox 0]/$mainCanvasWidth-0.01]
    $canvas yview moveto [expr [lindex $bbox 1]/$mainCanvasHeight-0.01]
    set preLoadModules $Subnet(Subnet${subnet}_Modules)
    set inserting 1
    loadnet $netedit_loadnet
    set inserting 0
    unselectAll
    foreach module $Subnet(Subnet${subnet}_Modules) {
	if { [lsearch $preLoadModules $module] == -1 } {
	    $module addSelected
	}
    }
    
    set NetworkChanged 1

}

proc subnet_bbox { subnet { cheat 1} } {
    global Subnet
    return [compute_bbox $Subnet(Subnet${subnet}_canvas) \
		$Subnet(Subnet${subnet}_Modules) 1]
}

proc compute_bbox { canvas { items "" } { cheat 0 } } {
    set canvasbounds [$canvas cget -scrollregion]
    set maxx [lindex $canvasbounds 0]
    set maxy [lindex $canvasbounds 1]
    set minx [lindex $canvasbounds 2]
    set miny [lindex $canvasbounds 3]
    global CurrentlySelectedModules
    if { $items == ""} { set items $CurrentlySelectedModules }
    if { ![llength $items] } { return [list 0 0 0 0] }
    foreach item $items {
	set bbox [$canvas bbox $item]
	if { $cheat && [lsearch [$canvas gettags $item] module] != -1 } {
	    if { [expr [lindex $bbox 2] - [lindex $bbox 0] < 10]  } {
		set bbox [lreplace $bbox 2 2 [expr [lindex $bbox 0]+180]]
	    }
	    if { [expr [lindex $bbox 3] - [lindex $bbox 1] < 10]  } {
		set bbox [lreplace $bbox 3 3 [expr [lindex $bbox 1]+80]]
	    }
	}
	if { [lindex $bbox 0] <= $minx} { set minx [lindex $bbox 0] }
	if { [lindex $bbox 1] <= $miny} { set miny [lindex $bbox 1] }
	if { [lindex $bbox 2] >  $maxx} { set maxx [lindex $bbox 2] }
	if { [lindex $bbox 3] >  $maxy} { set maxy [lindex $bbox 3] }
    }
    return [list $minx $miny $maxx $maxy]
}

proc popupLoadMenu {} {
    global NetworkChanged
    if $NetworkChanged {
	set result [tk_messageBox -type yesnocancel -parent . -title "Warning" \
			-message "Your network has not been saved.\n\nWould you like to save before loading a new one?" -icon warning ]
	if {![string compare "yes" $result]} { popupSaveMenu }
	if {![string compare "cancel" $result]} { return }
    }

    set types {
	{{SCIRun Net} {.net} }
	{{Uintah Script} {.uin} }
	{{Dataflow Script} {.sr} }
	{{Other} { * } }
    } 
    
    set netedit_loadnet [tk_getOpenFile -filetypes $types ]
    if { $netedit_loadnet == ""} return
    #dont ask user before clearing canvas
    ClearCanvas 0
    loadnet $netedit_loadnet
}

proc ClearCanvas { {confirm 1} {subnet 0} } {
    # destroy all modules
    global NetworkChanged
    if !$NetworkChanged { set confirm 0 }
    set result "ok"    
    if { $confirm } {
	set result \
	    [tk_messageBox -title "Warning" -type yesno -parent . -icon warning -message \
		 "Your network has not been saved.\n\nAll modules and connections will be deleted.\n\nReally clear?"]	
    }
    if {!$confirm || [string compare "yes" $result] == 0} {
	global Subnet netedit_savefile CurrentlySelectedModules
	foreach module $Subnet(Subnet${subnet}_Modules) {
	    if { [string first Render_Viewer $module] != -1 } {
		moduleDestroy $module
	    }
	}
	foreach module $Subnet(Subnet${subnet}_Modules) {
	    moduleDestroy $module
	}

	wm title . "SCIRun" ;# Reset Main Window Title
	set netedit_savefile ""
	set CurrentlySelectedModules ""
	set NetworkChanged 0
    }   
}

proc NiceQuit {} {
    global NetworkChanged netedit_savefile
    if {$NetworkChanged} {
        if {[winfo exists .standalone] } {
	    set result [createSciDialog -warning -title "Quit?" -button1 "Don't Save" -button2 "Cancel" -button3 "Save" \
                           -message "Your session has not been saved.\nWould you like to save before exiting?"  ]
	    if {![string compare "-1" $result]} { return }
	    if {![string compare "2" $result]} { return }
	    if {![string compare "3" $result]} { app save_session }
	} else {
	    set result [createSciDialog -warning -title "Quit?" -button1 "Don't Save" -button2 "Cancel" -button3 "Save" \
                           -message "Your network has not been saved.\nWould you like to save before exiting?" ]
	    if {![string compare "-1" $result]} { return }
	    if {![string compare "2" $result]} { return }
	    if {![string compare "3" $result]} { 
		puts -nonewline "Saving $netedit_savefile..."
		popupSaveMenu
	    }	
	}
    } 
    puts "Goodbye!"
    netedit quit
}

proc initInfo { {force 0 } } {
    global userName runDate runTime notes env
    if { $force || ![info exists userName] } {
	if { [info exists env(LOGNAME)] } { set userName $env(LOGNAME) 
	} elseif [info exists env(USER)] { set userName $env(USER) 
	} else { set userName unknown }
    }
    if { $force || ![info exists runDate] } {
	set runDate [clock format [clock seconds] -format "%a %b %d %Y"]
    }
    if { $force || ![info exists runTime] } {
	set runTime [clock format [clock seconds] -format "%H:%M:%S"]
    }
    if { ![info exists notes] } { set notes "" }    
}    

# This proc was added by Mohamed Dekhil to save some info about the net
proc popupInfoMenu {} {
    global userName runDate runTime notes
    global oldUserName oldRunDate oldRunTime oldNotes
    initInfo

    set oldUserName $userName
    set oldRunDate $runDate
    set oldRunTime $runTime
    set oldNotes $notes

    set w .netedit_info
    if {[winfo exists $w]} {
	raise $w
	return
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

    button $w.fdt.reset -text "Reset Date & Time" -command "initInfo 1"

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
    pack $w.fdt.reset -side right

    pack $w.fnotes.lnotes $w.fnotes.tnotes -side left
    pack $w.fnotes.scroll -side right -fill y

    pack $w.fbuttons.ok $w.fbuttons.clear $w.fbuttons.cancel -side right -padx 5 -pady 5 -ipadx 3 -ipady 3
}

proc infoClear {w} {
    global userName runDate runTime notes
    set userName ""
    set runDate ""
    set runTime ""
    set notes ""
    $w.fnotes.tnotes delete 1.0 end
}

proc infoOk {w} {
    global notes
    set notes [$w.fnotes.tnotes get 1.0 end]
    NetworkChanged
    destroy $w
}

proc infoCancel {w} {
    global userName runDate runTime notes
    global oldUserName oldRunDate oldRunTime oldNotes
    set userName $oldUserName
    set runDate $oldRunDate
    set runTime $oldRunTime
    set notes $oldNotes
    destroy $w
} 

proc loadfile {netedit_loadfile} {
    puts "NOTICE: `loadfile' has been disabled."
    puts "   To use old nets, remove the `loadfile' and `return' lines"
    puts "   from near the top of the file."
    return
}

proc loadnet { netedit_loadfile } {
    # Check to see of the file exists; warn user if it doesnt
    if { ![file exists $netedit_loadfile] } {
	createSciDialog -warning -message 
	    "File \"$netedit_loadfile\" does not exist."
	return
    }

    global netedit_loadfile_global
    set netedit_loadfile_global $netedit_loadfile

    global netedit_savefile NetworkChanged Subnet inserting
    if { !$inserting || ![string length $netedit_savefile] } {
	# Cut off the path from the net name and put in on the title bar:
	set name [lindex [file split "$netedit_loadfile"] end]
	wm title . "SCIRun ($name)"
	# Remember the name of this net for future "Saves".
	set netedit_savefile $netedit_loadfile
    }

    # The '#' below is not a comment... This souces the network file globally
    uplevel \#0 {source $netedit_loadfile_global}
    set Subnet(Subnet$Subnet(Loading)_filename) $netedit_loadfile
    if { !$inserting } { set NetworkChanged 0 }
}

# Ask the user to select a data directory 
# (Because the enviroment variable SCIRUN_DATA was not set)
proc getDataDirectory { dataset } {
   set answer [createSciDialog -warning -button1 "Ok" -button2 "Quit SCIRun" -message \
         "The '$dataset' dataset was specified (either by the enviroment variable\nSCIRUN_DATASET or by the network loaded).  However, the location of\nthis dataset was not specified (with the SCIRUN_DATA env var).  Please\nselect a directory (eg: /usr/sci/data/SCIRunData/1.20.0).  Note, this directory\nmust have the '$dataset' subdirectory in it." ]
   case $answer {
       1 "return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.20.0]"
       2 "netedit quit"
   }
}

#
# Tell the user the reason that they are being asked for the data, and
# then ask for the data.  If "warn_user" is "true", warning is displayed,
# otherwise we bring up the choose directory dialog directly.  "warn_user"
# most likely should only be displayed the first time.
#
proc getSettingsDirectory { warn_user } {

   if { "$warn_user" == "true" } {
      set answer [createSciDialog -warning -button1 "Ok" -parent . -button2 "Quit SCIRun" -message \
         "The enviroment variables SCIRUN_DATA and/or SCIRUN_DATASET\nare not set (or are invalid).  You must specify a valid data\nset directory in order to use this net!  You will now be asked\nto select the directory of the dataset you are interested in.\n(eg: /usr/sci/data/SCIRunData/1.20.0/sphere)\n\nFYI, if you set these environment variables, you will not need\nto select a directory manually when you load this network." ]
       case $answer {
	   1 "return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.20.0]"
	   2 "netedit quit"
       }
   }
   return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.20.0]
}

#
# Verify that the "file_name" file exists.  If not, tells the user that
# they will next have to input valid data.  It is up to the calling routine
# to check if verifyFile() returns "true", and if not, to ask the user for
# new information.
#
proc verifyFile { file_name } {
  if {![file isfile $file_name]} {

     set message "Error: $file_name is not a valid file.\nPlease select a valid directory (eg: /usr/sci/data/SCIRunData/1.20.0/sphere)"

     # This occurs if the user presses "cancel".
     if { "$file_name" == "//.settings" } {
        set message "You must select a data set directory for use with this network.\n\nEg: /usr/sci/data/SCIRunData/1.20.0/sphere"
     }

     createSciDialog -error -message "$message"

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

   if { "$DATASET" == "" } {
       # if env var SCIRUN_DATASET not set... default to sphere:
       set DATASET "sphere"
   } else {
       if { "$DATADIR" == 0 } {
          # DATASET specified, but not DATADIR.  Ask for DATADIR
          set DATADIR [getDataDirectory $DATASET]
	  # Push out to the environment so user doesn't get asked
	  # again if we need to check for this var again.
	  # (Do it twice do to tcl bug...)
	  array set env "SCIRUN_DATA    $DATADIR"
	  array set env "SCIRUN_DATA    $DATADIR"
       }
   }

   if { "$DATADIR" != "" && \
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

	 # Push out to the environment so user doesn't get asked
	 # again if we need to check for these vars again.
	 # NOTE: For some reason you have to do this twice... perhaps
	 # a newer version of TCL will fix this...
	 array set env [list SCIRUN_DATA $DATADIR SCIRUN_DATASET $DATASET]
	 array set env [list SCIRUN_DATA $DATADIR SCIRUN_DATASET $DATASET]
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

# if reset == "true" then remove the progress buttons so that
# when it is brought up by the "About" menu, it will only have
# the "ok" button.
proc hideSplash { reset } {
    wm withdraw .splash
    if { $reset == "true" } {
	destroy .splash.fb
	button .splash.ok -text "OK" -width 10 -command "wm withdraw .splash"
	pack .splash.ok -side bottom -padx 5 -pady 5 -fill none
    }
}

proc showSplash { imgname {steps none} } {
    global SCIRUN_SRCDIR

    if {[winfo exists .splash]} {
	# Center on main SCIRun window
        wm geometry .splash +[expr 135+[winfo x .]]+[expr 170+[winfo y .]]
	if { [winfo ismapped .splash] == 1} {
	    raise .splash
	} else {
	    wm deiconify .splash
	}
	return
    }

    set filename [file join $SCIRUN_SRCDIR $imgname]
    image create photo ::img::splash -file "$filename"
    toplevel .splash

    # Center splash in main SCIRun window:
    #
    # Must do it this way as "." isn't positioned at this point...({}
    # delays the execution of the winfo command.)
    #
    after 0 {wm geometry .splash +[expr 135+[winfo x .]]+[expr 170+[winfo y .]]}

    wm protocol .splash WM_DELETE_WINDOW "wm withdraw .splash"

    wm title .splash {Welcome to SCIRun}
    label .splash.splash -image ::img::splash
    pack .splash.splash
    if { ![string equal $steps none ] } {
	iwidgets::feedback .splash.fb -steps $steps -labeltext \
	    "{Loading package:                 }"
	pack .splash.fb -padx 5 -fill x
	# The following line forces the window to be the correct size... this is a
	# hack to fix things on the mac. -Dav
	wm geometry .splash "" 
    }
    update idletasks
}

global LicenseResult
set licenseResult decline


proc licenseDialog { {firsttime 0} } {
    if $firsttime { return "accept" }
    global SCIRUN_SRCDIR licenseResult userData
    set filename [file join $SCIRUN_SRCDIR LICENSE]
    set stream [open $filename r]
    toplevel .license

    wm title .license {UNIVERSITY OF UTAH RESEARCH FOUNDATION PUBLIC LICENSE}

    frame .license.text -borderwidth 1 -class Scroll -highlightthickness 1 \
	-relief sunken -takefocus 0
    text .license.text.text -wrap word  -borderwidth 2 -relief sunken \
	-yscrollcommand ".license.text.y set" -width 80 -height 20
    scrollbar .license.text.y -borderwidth 2 -elementborderwidth 1 \
	-orient vertical -takefocus 0 -highlightthicknes 0 \
	-command ".license.text.text yview"
    grid columnconfigure .license.text 0 -weight 1
    grid rowconfigure .license.text    0 -weight 1 -pad 2
    grid .license.text.text -column 0 -row 0 -sticky news
    grid .license.text.y -column 1 -row 0 -sticky news

    pack .license.text -expand 1 -fill both -side top
    while { ![eof $stream] } {
	gets $stream line
	.license.text.text insert end "$line\n"
    }
    close $stream
    .license.text.text configure -state disabled
    bind .license.text.text <1> {focus %W}
    frame .license.b
    pack .license.b -side bottom
    set result decline
    set userData(first) ""
    set userData(last) ""
    set userData(email) ""
    set userData(aff) ""
    set licenseResult decline
    if { $firsttime } {
	frame .license.b.entry
	
	set w .license.b.entry.first
	frame $w
	label $w.lab -justify right -text "First Name:"
	entry $w.entry -width 35 -textvariable userData(first)
	pack $w.entry $w.lab -side right -expand 0

	set w .license.b.entry.last
	frame $w
	label $w.lab -justify right -text "Last Name:"
	entry $w.entry -width 35 -textvariable userData(last)
	pack $w.entry $w.lab -side right -expand 0

	set w .license.b.entry.email
	frame $w
	label $w.lab -justify right -text "E-Mail Address:"
	entry $w.entry -width 35 -textvariable userData(email)
	pack $w.entry $w.lab -side right -expand 0

	set w .license.b.entry.affil
	frame $w
	label $w.lab -justify right -text "Affiliation:"
	entry $w.entry -width 35 -textvariable userData(aff)
	pack $w.entry $w.lab -side right -expand 0

	set w .license.b.entry
	pack $w.first $w.last $w.email $w.affil -side top -expand 1 -fill x

	set w .license.b.buttons
	frame $w
	button $w.accept -text Accept -width 12 -command "licenseAccept"
	button $w.decline -text Decline -width 12 \
	    -command "set licenseResult cancel
                      catch {destroy .license}"
	button $w.later -text Later -width 12\
	    -command "set licenseResult later
                      catch {destroy .license}"
	pack $w.accept $w.decline $w.later -padx 5 -pady 5 -side left -padx 20
	pack .license.b.buttons  .license.b.entry -side bottom

	wm protocol .license WM_DELETE_WINDOW {
	    createSciDialog -error -message "You must choose Accept, Decline, or Later to continue."
	}

    } else {
        wm protocol .license WM_DELETE_WINDOW { destroy .license }
	button .license.b.ok -text "OK" -width 10 -command {destroy .license}
	pack .license.b.ok -padx 2 -pady 2 -side bottom
    }
    moveToCursor .license
    wm deiconify .license
    grab .license
    if { $firsttime } { tkwait window .license }
    return $licenseResult
}

proc licenseAccept { } {
    set ouremail "scirun-register@sci.utah.edu"
    set majordomo "majordomo@sci.utah.edu"
    global licenseResult userData SCIRun_version
    if { [string length $userData(first)] &&
	 [string length $userData(last)] &&
	 [string length $userData(email)] } {

	set w .license.confirm
	toplevel $w
	wm title $w "Confirm E-Mail"
	wm geometry $w +180+240
	wm withdraw $w

	label $w.label -justify left -text "The following e-mail will be sent:"
	pack $w.label -side top -expand 1 -fill x
	text $w.text -wrap word -height 10 -width 50  -borderwidth 1
	$w.text insert end "To: $ouremail\n"
	$w.text insert end "Subject: SCIRun_Registration_Notice\n\n"
	$w.text insert end "The following e-mail was automatically generated by SCIrun $SCIRun_version\n\n"
	$w.text insert end "First Name:  $userData(first)\n"
	$w.text insert end "Last Name:  $userData(last)\n"
	$w.text insert end "E-Mail:  $userData(email)\n"
	$w.text insert end "Affiliation:  $userData(aff)\n"
	pack $w.text -expand 1 -fill both -side top
	set check 1
	checkbutton $w.check -variable check -text  \
	    "Register \($userData(email)\) for the SCIRun Users' Mailing List"
	$w.check select

	frame $w.b
	button $w.b.cancel -text Cancel -width 12 \
	    -command "catch {destroy .license.confirm}"
	button $w.b.ok -text OK -width 12 -command \
	    "set licenseResult accept; catch {destroy .license.confirm}"
	pack $w.b.ok $w.b.cancel -padx 5 -pady 5 -side left -padx 20
	pack $w.b $w.check -side bottom 

	grab $w
	wm deiconify $w
	raise $w
	wm withdraw .license
	tkwait window $w

	if ![string equal $licenseResult accept] {
	    wm deiconify .license
	} else {
	    catch {destroy .license}
	    set name [file join / tmp SCIRun.Register.email.txt]
	    catch "set out \[open $name w\]"
	    if { ![info exists out] } {
		puts "Cannot open $name for writing.  Giving up."
		return
	    }
	    puts $out "\nThe following e-mail was automatically generated by SCIRun $SCIRun_version"
	    puts $out "\nFirst Name:  $userData(first)"
	    puts $out "Last Name:  $userData(last)"
	    puts $out "E-Mail:  $userData(email)"
	    puts $out "Affiliation:  $userData(aff)"
	    catch "close $out"
	    netedit sci_system /usr/bin/Mail -n -s SCIRun_Registration_Notice $ouremail < $name
	    catch [file delete $name]

	    if { $check } {
		set name [file join / tmp SCIRun.Register.email.txt]
		catch "set out \[open $name w\]"
		if { ![info exists out] } {
		    puts "Cannot open $name for writing.  Giving up."
		    return
		}
		puts $out "subscribe scirun-users $userData(email)\n"
		catch "close $out"
		netedit sci_system /usr/bin/Mail -n $majordomo < $name
		catch [file delete $name]
	    }

	}

    } else {
	tk_messageBox -type ok -parent .license -icon error \
	    -message "You must enter a First Name, Last Name, and E-Mail address to Accept."
    }
}

proc validDir { name } {
    return [expr [file isdirectory $name] && \
		 [file writable $name] && [file readable $name]]
}

proc getOnTheFlyLibsDir {} {
    global env SCIRUN_OBJDIR tcl_platform
    set binOTF [file join $SCIRUN_OBJDIR on-the-fly-libs]
    set dir ""
    if [info exists env(SCIRUN_ON_THE_FLY_LIBS_DIR)] {
	set dir $env(SCIRUN_ON_THE_FLY_LIBS_DIR)
	catch "file mkdir $dir"
	if { [validDir $dir] && ![llength [glob -nocomplain -directory $dir *]] } {
	    foreach name [glob -nocomplain -directory $binOTF *.cc *.d *.o *.so] {
		file copy $name $dir
	    }
	}
		
	    
    }

    if ![validDir $dir] {
	set dir $binOTF
    }

    if ![validDir $dir] {
	set home [file nativename ~]
	set dir [file join $home SCIRun on-the-fly-libs $tcl_platform(os)]
	catch "file mkdir $dir"
	if { [validDir $dir] && ![llength [glob -nocomplain -directory $dir *]] } {
	    foreach name [glob -nocomplain -directory $binOTF *.cc *.d *.o *.so] {
		file copy $name $dir
	    }
	}
    }
    if { ![validDir $dir] } {
	tk_messageBox -type ok -parent . -icon error -message \
	    "SCIRun cannot find a directory to store dynamically compiled code.\n\nPlease quit and set the environment variable SCIRUN_ON_THE_FLY_LIBS_DIR to a readable and writable directory.\n\nDynamic code generation will not work.  If you continue this session, networks may not execute correctly."
	return $binOTF
    }

    set makefile [file join $SCIRUN_OBJDIR on-the-fly-libs Makefile]
    if [catch "file copy -force $makefile $dir"] {
	tk_messageBox -type ok -parent . -icon error -message \
	    "SCIRun cannot copy $makefile to $dir.\n\nThe Makefile was generated during configure and is necessasary for dynamic compilation to work.  Please reconfigure SCIRun to re-generate this file.\n\nDynamic code generation will not work.  If you continue this session, networks may not execute correctly."
	return $binOTF
    }
	
    return $dir
}



# Removes the element at pos from a list without a set - similar to lappend
# ex: 
#   set bob "0 1 2 3 4 5"
#   listRemove bob 2
#   puts $bob
# output:
#   0 1 3 4 5
proc listRemove { name pos } {
    uplevel 1 set $name \[list [lreplace [uplevel 1 set $name] $pos $pos]\]
}

# Finds then removes an element from a list without a set - similar to lappend
# ex: 
#   set bob "foo bar foo2 bar2 foo3 bar3"
#   listFindAndRemove bob foo2
#   puts $bob
# output:
#   foo bar bar2 foo3 bar3
proc listFindAndRemove { name elem } {
    set elements [uplevel 1 set $name]
    set pos [lsearch $elements $elem]    
    uplevel 1 set $name \[list [lreplace $elements $pos $pos]\]
}


proc initVar { var } {
    if { [string first msgStream $var] != -1 } return
    uplevel \#0 trace variable \"$var\" w networkHasChanged
}


# Debug procedure to print global variable values
proc printvars { pattern } {
    foreach name [lsort [uplevel \#0 "info vars *${pattern}*"]] { 
	upvar \#0 $name var
	puts "set \"$name\" \{$var\}"
    }
}
