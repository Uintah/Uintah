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
global minicanvs
set minicanvas ".top.globalViewFrame.canvas"
global Subnet
set Subnet(Subnet0_minicanvas) $minicanvas

global inserting
set inserting 0

global insertPosition
set insertPosition 0

global netedit_savefile
set netedit_savefile ""

global NetworkChanged
set NetworkChanged 0



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
    if 0 {
	.main_menu.file.menu add command -label "Save Postscript..." -underline 0 \
	    -command ".bot.neteditFrame.canvas postscript -file /tmp/canvas.ps -x 0 -y 0 -width 4500 -height 4500" -state disabled
    }
    .main_menu.file.menu add command -label "Execute All" -underline 0 \
	-command "netedit schedule" -state disabled
    .main_menu.file.menu add cascade -label "New" -underline 0\
        -menu .main_menu.file.menu.new -state disabled

    # This was added by Mohamed Dekhil to add some infor to the net
    .main_menu.file.menu add command -label "Add Info..." -underline 0 \
	-command "popupInfoMenu"

    .main_menu.file.menu add command -label "Quit" -underline 0 \
	    -command "NiceQuit"


    pack .main_menu.file -side left
    global ToolTipText
    Tooltip .main_menu.file $ToolTipText(FileMenu)
    
    menubutton .main_menu.help -text "Help" -underline 0 \
	-menu .main_menu.help.menu -direction below
    menu .main_menu.help.menu -tearoff false
    .main_menu.help.menu add checkbutton -label "Show Tooltips" -underline 0 \
	-variable tooltipsOn

    .main_menu.help.menu add command -label "About..." -underline 0 \
	-command  "showSplash"
    .main_menu.help.menu add command -label "License..." -underline 0 \
	-command  "licenseDialog"

    pack .main_menu.help -side right
    
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

    menu $maincanvas.modulesMenu -tearoff false

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
    bind all <Control-y> "redo"

    setupMainSubnet
        
}

proc canvasScroll { canvas { dx 0.0 } { dy 0.0 } } {
    if {$dx!=0.0} {$canvas xview moveto [expr $dx+[lindex [$canvas xview] 0]]}
    if {$dy!=0.0} {$canvas yview moveto [expr $dy+[lindex [$canvas yview] 0]]}
}

# Activate the "File" menu items - called from C after all packages are loaded
proc activate_file_submenus { } {
    .main_menu.file.menu entryconfig 0 -state active
    .main_menu.file.menu entryconfig 1 -state active
    .main_menu.file.menu entryconfig 2 -state active
    .main_menu.file.menu entryconfig 3 -state active
    .main_menu.file.menu entryconfig 4 -state active
    .main_menu.file.menu entryconfig 5 -state active
    .main_menu.file.menu entryconfig 6 -state active
    .main_menu.file.menu entryconfig 7 -state active
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
    pack .main_menu.$pack -side left
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
    update idletasks
}

proc createModulesMenu { subnet } {
    global ModuleMenu Subnet
    set canvas $Subnet(Subnet${subnet}_canvas)
    if { ![info exists ModuleMenu] || \
	  [info exists ModuleMenu($canvas)] } { return }
    set ModuleMenu($canvas) 1
    foreach pack $ModuleMenu(packages) {
	# Add a separator to the right-button menu for this package if this
	# isn't the first package to go in there
	if { [$canvas.modulesMenu index end] != "none" } \
	    { $canvas.modulesMenu add separator }
	foreach cat $ModuleMenu(${pack}_categories) {
	    # Add the category to the right-button menu
	    $canvas.modulesMenu add cascade -label "$ModuleMenu($cat)" \
		-menu $canvas.modulesMenu.$cat
	    menu $canvas.modulesMenu.$cat -tearoff false
	    foreach mod $ModuleMenu(${pack}_${cat}_modules) {
		$canvas.modulesMenu.$cat add command \
		    -label "$ModuleMenu($mod)" \
		    -command "global Subnet; set Subnet(Loading) $subnet; addModuleAtMouse \"$ModuleMenu($pack)\" \"$ModuleMenu($cat)\" \"$ModuleMenu($mod)\"; set Subnet(Loading) 0"
	    }
	}
    }
    global SCIRUN_SRCDIR
    set filelist1 [glob -nocomplain -dir $SCIRUN_SRCDIR/Subnets *.net]
    set filelist2 [glob -nocomplain -dir ~/Subnets *.net]
    set subnetfiles [concat $filelist1 $filelist2]
    if [llength $subnetfiles] {
	if { [$canvas.modulesMenu index end] != "none" } \
	    { $canvas.modulesMenu add separator }
	# Add the category to the right-button menu
	$canvas.modulesMenu add cascade -label "Sub-Networks" \
		-menu $canvas.modulesMenu.subnet
	menu $canvas.modulesMenu.subnet -tearoff false
	foreach file $subnetfiles {
	    set name [join [lrange [split [lindex [split $file "/"] end] "."] 0 end-1] "."]
	    $canvas.modulesMenu.subnet add command \
		-label "$name" \
		-command "loadSubnet $subnet {$file}"
	}
    }
	
    update idletasks
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
    global inserting insertPosition Subnet
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
    if { $inserting && !$insertPosition } {
	global modulesBbox
	set xpos [expr int([expr $xpos+[lindex $modulesBbox 2]])]
	set ypos [expr $ypos+[$canvas canvasy 0]]
    } else { ;# create the module relative to current screen position
	set xpos [expr $xpos+[$canvas canvasx 0]]
	set ypos [expr $ypos+[$canvas canvasy 0]]
    }
    $modid make_icon $xpos $ypos
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
	netedit savenetwork $netedit_savefile
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
	netedit savenetwork $netedit_savefile
	set NetworkChanged 0
	# Cut off the path from the net name and put in on the title bar:
	wm title . "SCIRun ([lindex [split "$netedit_savefile" /] end])"
    }
}

proc popupInsertMenu { {subnet 0} } {
    global inserting insertPosition Subnet NetworkChanged
    set inserting 1
    
    #get the net to be inserted
    set types {
	{{SCIRun Net} {.net} }
	{{Uintah Script} {.uin} }
	{{Dataflow Script} {.sr} }
	{{Other} { * } }
    } 
    set netedit_loadnet [tk_getOpenFile -filetypes $types ]
    if { $netedit_loadnet == "" || ![file exists $netedit_loadnet]} { 
	set inserting 0
	return
    }
    
    set canvas $Subnet(Subnet${subnet}_canvas)    
    set insertBbox [$canas cget -scrollregion]
    set bbox $insertBbox
    # get the bbox for the net being inserted by
    # parsing netedit_loadnet for bbox 

    set fchannel [open $netedit_loadnet]
    set curr_line ""
    set curr_line [gets $fchannel]
    while { ![eof $fchannel] } {
	if { [string match "set bbox*" $curr_line] } {
	    eval $curr_line
	    set insertBbox $bbox
	    break
	}
	set curr_line [gets $fchannel]
    }

    set insertPosition 1
    set modules $Subnet(Subnet${subnet}_Modules)
    if { [llength $modules] } {
	set startX [expr [$canvas canvasx 0]+[lindex $insertBbox 0]]
	set startY [expr [$canvas canvasy 0]+[lindex $insertBbox 1]]
	set endX   [expr [$canvas canvasx 0]+[lindex $insertBbox 2]]
	set endY   [expr [$canvas canvasy 0]+[lindex $insertBbox 3]]
	foreach m $modules {
	    set curr_coords [$canvas coords $m]
	    if { [lindex $curr_coords 0] < $endX && \
		 [lindex $curr_coords 1] > $startX && \
		 [lindex $curr_coords 1] < $endY && \
		 [lindex $curr_coords 1] > $startY } {
		set insertPosition 0
		break
	    }
	}
    }
    global modulesBbox
    set modulesBbox [compute_bbox $canvas [$canvas find withtag "module"]]
    loadnet $netedit_loadnet
    set NetworkChanged 1
    set inserting 0
}

proc compute_bbox { canvas { items "" } } {
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
	set result [tk_messageBox -type yesnocancel -parent . -message \
			"Your network has not been saved.\nWould you like to save before loading a new one?" -icon warning ]
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
    set result "ok"
    if { $confirm && $NetworkChanged } {
	set result \
	    [tk_messageBox -type okcancel -parent . -icon warning -message \
		 "Your network has not been saved.\nALL modules and connections will be cleared.\nReally clear?"]	
    }
    if {!$confirm || [string compare "ok" $result] == 0} {
	global Subnet netedit_savefile CurrentlySelectedModules
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
	    set result [tk_messageBox -type yesnocancel -parent . -message \
			    "Your session has not been saved.\nWould you like to save before exiting?" -icon warning ]
	    if {![string compare "yes" $result]} { app save_session }
	    if {![string compare "cancel" $result]} { return }
	} else {
	    set result [tk_messageBox -type yesnocancel -parent . -message \
			    "Your network has not been saved.\nWould you like to save before exiting?" -icon warning ]
	    if {![string compare "cancel" $result]} { return }
	    if {![string compare "yes" $result]} { 
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

proc loadnet {netedit_loadfile } {
    # Check to see of the file exists; warn user if it doesnt
    if { ![file exists $netedit_loadfile] } {
	tk_messageBox -type ok -parent . -icon warning -message \
	    "File \"$netedit_loadfile\" does not exist."
	return
    }
    # Cut off the path from the net name and put in on the title bar:
    wm title . "SCIRun ([lindex [split "$netedit_loadfile" / ] end])"
    # Remember the name of this net for future "Saves".
    global netedit_savefile NetworkChanged Subnet
    set netedit_savefile $netedit_loadfile
    # The '#' below is not a comment...
    uplevel #0 {source $netedit_savefile}
    set Subnet(Subnet$Subnet(Loading)_filename) $netedit_loadfile
    set NetworkChanged 0
}

# Ask the user to select a data directory 
# (Because the enviroment variable SCIRUN_DATA was not set)
proc getDataDirectory { dataset } {
   set answer [tk_messageBox -type okcancel -parent . -message \
         "The '$dataset' dataset was specified (either by the enviroment variable SCIRUN_DATASET or by the network loaded).  However, the location of this dataset was not specified (with the SCIRUN_DATA env var).  Please select a directory (eg: /usr/sci/data/SCIRunData/1.10.0).  Note, this directory must have the '$dataset' subdirectory in it." ]
   case $answer {
       ok "return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.10.0]"
       cancel "netedit quit"
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
      set answer [tk_messageBox -type okcancel -parent . -message \
         "The enviroment variables SCIRUN_DATA and/or SCIRUN_DATASET are not set (or are invalid).  You must specify a valid data set directory in order to use this net!  You will now be asked to select the directory of the dataset you are interested in.  (eg: /usr/sci/data/SCIRunData/1.10.0/sphere) (FYI, if you set these environment variables, you will not need to select a directory manually when you load this network.)" ]
       case $answer {
	   ok "return [tk_chooseDirectory -mustexist true -initialdir /usr/sci/data/SCIRunData/1.10.0]"
	   cancel "netedit quit"
       }
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

proc showSpash { steps image_file } {
    showSplash $steps
}

proc showSplash { {steps none} } {
    global SCIRUN_SRCDIR
    set filename [file join $SCIRUN_SRCDIR main scisplash.ppm]
    toplevel .loading
    wm geometry .loading 504x482+135+170
    wm title .loading {Welcome to SCIRun}
    update idletasks
    image create photo ::img::splash -file "$filename"
    label .loading.splash -image ::img::splash
    pack .loading.splash
    if { ![string equal $steps none ] } {
	iwidgets::feedback .loading.fb -steps $steps -labeltext \
	    "{Loading package:                 }"
	pack .loading.fb -padx 5 -fill x
    } else {
	button .loading.ok -text "OK" \
	    -command "destroy .loading"
	pack .loading.ok -side bottom -padx 5 -pady 5 -fill none
    }

    update idletasks
}

proc licenseDialog { {firsttime 1} } {
    global SCIRUN_SRCDIR
    set filename [file join $SCIRUN_SRCDIR LICENSE]
    set stream [open $filename r]
    toplevel .license
    wm geometry .license 504x482+135+170
    wm title .license {UNIVERSITY OF UTAH RESEARCH FOUNDATION PUBLIC LICENSE}
    frame .license.text -borderwidth 1 -class Scroll -highlightthickness 1 \
	-relief sunken -takefocus 0
    text .license.text.text -wrap word  -borderwidth 0 -relief flat \
	-yscrollcommand ".license.text.y set"
    scrollbar .license.text.y -borderwidth 0 -elementborderwidth 1 \
	-orient vertical -takefocus 0 -highlightthicknes 0 \
	-command ".license.text.text yview"
    grid columnconfigure .license.text 0 -weight 1
    grid rowconfigure .license.text    0 -weight 1
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
    if { $firsttime } {

	button .license.b.accept -text Accept -command {destroy .license}
	button .license.b.decline -text Decline -command {destroy .license}
	pack .license.b.accept .license.b.decline -padx 5 -pady 5 -side right
    } else {
	button .license.b.OK -text OK -command {destroy .license}
	pack .license.b.OK -padx 5 -pady 5 -side bottom
    }
    raise .license
    grab .license
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
