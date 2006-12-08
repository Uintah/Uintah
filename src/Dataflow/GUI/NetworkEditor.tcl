#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
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

source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/defaults.tcl
source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/Module.tcl
source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/Connection.tcl
source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/Port.tcl
source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/Subnet.tcl
source [netedit getenv SCIRUN_SRCDIR]/Dataflow/GUI/UIvar.tcl
source [netedit getenv SCIRUN_SRCDIR]/Core/GUI/Range.tcl

set SCIRUN_SRCDIR [netedit getenv SCIRUN_SRCDIR]
set smallIcon [image create photo -file "$SCIRUN_SRCDIR/pixmaps/scirun-icon-small.ppm"]
set splashImageFile "$SCIRUN_SRCDIR/main/scisplash.ppm"
set bioTensorSplashImageFile "$SCIRUN_SRCDIR/Packages/Teem/Dataflow/GUI/splash-tensor.ppm"
set bioFEMSplashImageFile "$SCIRUN_SRCDIR/Packages/BioPSE/Dataflow/GUI/splash-biofem.ppm"
set bioImageSplashImageFile "$SCIRUN_SRCDIR/Packages/Teem/Dataflow/GUI/splash-bioimage.ppm"
set fusionViewerSplashImageFile "$SCIRUN_SRCDIR/Packages/Fusion/Dataflow/GUI/splash-fusionviewer.ppm"
set levelSetSegmenterViewerSplashImageFile "$SCIRUN_SRCDIR/main/scisplash.ppm"

set modname_font "-Adobe-Helvetica-Bold-R-Normal-*-12-120-75-*"
set ui_font "-Adobe-Helvetica-Medium-R-Normal-*-12-120-75-*"
set time_font "-Adobe-Courier-Medium-R-Normal-*-12-120-75-*"

set firstIcon 1

set mainCanvasWidth    4500.0
set mainCanvasHeight   4500.0
set maincanvas .topbot.pane1.childsite.frame.canvas
set minicanvas .topbot.pane0.childsite.panes.pane0.childsite.pad.frame.canvas

# Records mouse position at button press to bring up menus at 
set mouseX 0
set mouseY 0

set Subnet(Subnet0_minicanvas) $minicanvas
set Subnet(Subnet0_canvas) $maincanvas
set Subnet(Subnet0_Name) Main
set Subnet(Subnet0_Modules) ""
set Subnet(Subnet0_connections) ""
set Subnet(Subnet0) 0
set Subnet(Loading) 0
set Subnet(num) 0

set inserting 0
set netedit_savefile ""
set NetworkChanged 0
set CurrentlySelectedModules ""
set disable_network_locking "0"
set network_executing "0"
trace variable network_executing w handle_network_executing

proc set_network_executing {val} {
    global network_executing disable_network_locking
    if { $disable_network_locking == "1" } {
	set network_executing "0"
	return
    }
    set network_executing $val
}

proc restore_not_executing_interface {} {
    global network_executing maincanvas Color disable_network_locking
     $maincanvas itemconfigure bgRect -fill $Color(NetworkEditor) \
	-outline $Color(NetworkEditor)
    
    bind . <Control-d> "moduleDestroySelected"
    bind . <Control-l> "ClearCanvas"
    bind . <Control-z> "undo"
    bind . <Control-e> "netedit scheduleall"
    bind . <Control-y> "redo"
    bind . <Control-o> "popupLoadMenu"
    bind . <Control-s> "popupSaveMenu"
    bind all <Control-q> "NiceQuit"
    .main_menu.file.menu entryconfig  0 -state active
    .main_menu.file.menu entryconfig  1 -state active
    .main_menu.file.menu entryconfig  2 -state active
    .main_menu.file.menu entryconfig  3 -state active
    .main_menu.file.menu entryconfig  5 -state active
    .main_menu.file.menu entryconfig  6 -state active
    .main_menu.file.menu entryconfig  7 -state active
    .main_menu.file.menu entryconfig  9 -state active
    .main_menu.file.menu entryconfig 11 -state active
}

proc disable_netedit_locking {} {
    global disable_network_locking
    set disable_network_locking "1"
    restore_not_executing_interface
}

proc handle_network_executing { var op1 op2} {
    global network_executing maincanvas Color
    # unbind/rebind the keystroke commands that can change network state.
    if { $network_executing == "1" } {
	$maincanvas itemconfigure bgRect -fill $Color(NetworkEditorLocked) \
	    -outline $Color(NetworkEditorLocked)

	bind . <Control-u> "disable_netedit_locking"
	bind . <Control-d> ""
	bind . <Control-l> ""
	bind . <Control-z> ""
	bind . <Control-e> ""
	bind . <Control-y> ""
	bind . <Control-o> ""
	bind . <Control-s> ""
	bind all <Control-q> ""
	.main_menu.file.menu entryconfig  0 -state disabled
	.main_menu.file.menu entryconfig  1 -state disabled
	.main_menu.file.menu entryconfig  2 -state disabled
	.main_menu.file.menu entryconfig  3 -state disabled
	.main_menu.file.menu entryconfig  5 -state disabled
	.main_menu.file.menu entryconfig  6 -state disabled
	.main_menu.file.menu entryconfig  7 -state disabled
	.main_menu.file.menu entryconfig  9 -state disabled
	.main_menu.file.menu entryconfig 11 -state disabled

    } else {
	restore_not_executing_interface
    }
}

proc setIcons { { w . } { size small } } {
    global firstIcon tcl_platform
    set srcdir [netedit getenv SCIRUN_SRCDIR]
    if { [string length $srcdir] } {
	set bitmap $srcdir/pixmaps/scirun-icon-$size.xbm
	set inverted $srcdir/pixmaps/scirun-icon-$size-inverted.xbm
    if { $tcl_platform(os) == "Windows NT" && $size == "small" && $firstIcon == 1 } {
      # make all subsequent windows use this icon.  This prevents an annoying problem
      # where an empty window pops up and then fills up and moves to the proper location
	  wm iconbitmap $w -default @$inverted
	  set firstIcon 0
	} elseif {$tcl_platform(os) != "Windows NT"} {
	  wm iconbitmap $w @$inverted	
	}
	wm iconmask $w @$bitmap
    }
}


# envBool <variable name>
#
#   envBool will query the enviromnent varaible's value and return as a boolean
#   Usage example:  envBool SCIRUN_INSERT_NET_COPYRIGHT
#
#   Turns 'true/false', 'on/off', 'yes/no', '1/0' into '1/0' respectively
#   (Non-existent and empty  variables are treated as 'false'.)
#
#   This function is case insensitive.
#
proc envBool { var } {
    set val [netedit getenv $var] ; # returns blank string if variable not set
    if { [string equal $val ""] } {
	return 0; # blank value is taken to mean false
    }
    if { ![string is boolean $val] } {
	puts "TCL envBool: Cannot determine boolean value of env: $var=$val"
	return 1; # follows the C convention of any non-zero value equals true
    }
    return [string is true $val]
}

proc safeSetWindowGeometry { w geom } {
    set realgeom [split [wm geometry $w] +]
    set geom [split $geom +]
    set sizepos [lsearch $geom *x*]
    if { $sizepos == -1 } {
	set width [winfo width $w]
	set height [winfo height $w]
    } else {
	set size [split [lindex $geom $sizepos] x]
	set geom [lreplace $geom $sizepos $sizepos]
	set width [lindex $size 0]
	set height [lindex $size 1]
    }

    set xoff [lindex $realgeom 1]
    set yoff [lindex $realgeom 2]

    if { [llength $geom] } {
	set xoff [lindex $geom end-1]
	set yoff [lindex $geom end]
    }
    
    set xoff    [expr ($xoff < 0) ? 0 : $xoff]
    set yoff    [expr ($yoff < 0) ? 0 : $yoff]

#    set swidth  [expr [winfo screenwidth .]  - $xoff]
#    set sheight [expr [winfo screenheight .] - $yoff]
	
#    set width   [expr ($swidth  < $width)  ? $swidth  : $width]
#    set height  [expr ($sheight < $height) ? $sheight : $height]

    wm geometry $w ${width}x${height}+${xoff}+${yoff}
}

rename toplevel __TCL_toplevel__
proc toplevel { args } {
    set win [uplevel 1 __TCL_toplevel__ $args]
    setIcons [lindex $args 0]
    set varname [string range $win 3 end]-ui_geometry
    upvar \#0 $varname geometry
    if { [info exists geometry] } {
	safeSetWindowGeometry $win $geometry
    }
    return $win
}


proc geometryTrace { args } {
    global geometry Subnet
    if { [info exists geometry] } {
	set w .
	if { $Subnet(Loading) } {
	    set w .subnet$Subnet(Loading)
	}
	safeSetWindowGeometry $w $geometry
    }
}

proc makeNetworkEditor {} {

    wm protocol . WM_DELETE_WINDOW { NiceQuit }
    wm minsize . 100 100

    global geometry
    trace variable geometry w geometryTrace

    set geom [netedit getenv SCIRUN_GEOMETRY]
    if { [string length $geom] } {
	safeSetWindowGeometry . $geom
    } else {
	safeSetWindowGeometry . 800x800
    }

    wm title . "SCIRun v[netedit getenv SCIRUN_VERSION]"
    setIcons . large

    loadToolTipText

    frame .main_menu -relief raised -borderwidth 3
    pack .main_menu -fill x
    
    menubutton .main_menu.file -text "File" -underline 0 \
	-menu .main_menu.file.menu
    menu .main_menu.file.menu -tearoff false
    menu .main_menu.file.menu.new -tearoff false
    .main_menu.file.menu.new add command -label "Create Module Skeleton..." \
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
	-command "backupNetwork; updateRunDateAndTime 0; netedit scheduleall" -state disabled

    .main_menu.file.menu add separator
    .main_menu.file.menu add cascade -label "Wizards" -underline 0 \
        -menu .main_menu.file.menu.new -state disabled
    .main_menu.file.menu add separator

    # This was added by Mohamed Dekhil to add some infor to the net
    .main_menu.file.menu add command -label "Network Properties" -underline 0 \
	-command "popupInfoMenu"

    .main_menu.file.menu add separator
    .main_menu.file.menu add command -label "Quit" -underline 0 \
	    -command "NiceQuit"


    pack .main_menu.file -side left
    global ToolTipText
    Tooltip .main_menu.file $ToolTipText(FileMenu)
    
    menubutton .main_menu.subnet -text "Sub-Networks" -underline 0 \
	-menu .main_menu.subnet.menu -direction below 
    menu .main_menu.subnet.menu -tearoff false -postcommand createSubnetMenu
    pack .main_menu.subnet -side left


    menubutton .main_menu.help -text "Help" -underline 0 \
	-menu .main_menu.help.menu -direction below
    menu .main_menu.help.menu -tearoff false
    .main_menu.help.menu add checkbutton -label "Show Tooltips" -underline 0 \
	-variable tooltipsOn

    # Mac hack to fix size of 'About' window ... sigh... 
    .main_menu.help.menu add command -label "About..." -underline 0 \
	-state disabled -command  "showProgress 1 none 1"

    .main_menu.help.menu add command -label "License..." -underline 0 \
	-command  "licenseDialog" -state disabled

    pack .main_menu.help -side right
    Tooltip .main_menu.help $ToolTipText(HelpMenu)
    
    tk_menuBar .main_menu .main_menu.file

    global leftFrame rightFrame botFrame topFrame Color
    global maincanvas minicanvas mainCanvasHeight mainCanvasWidth
    iwidgets::panedwindow .topbot -orient horizontal -thickness 0 -sashwidth 5000 -sashheight 10 -sashindent 0 -sashborderwidth 0 -sashcursor sb_v_double_arrow
    pack .topbot -expand 1 -fill both -padx 0 -pady 0 -ipadx 0 -ipady 0
    .topbot add topFrame -margin 0 -minimum 0
    .topbot add botFrame -margin 5 -minimum 0
    set topFrame [.topbot childsite topFrame]
    set botFrame [.topbot childsite botFrame]

    iwidgets::panedwindow $topFrame.panes -orient vertical -height 100 -thickness 0 -sashheight 5000 -sashwidth 10 -sashindent 0 -sashborderwidth 0 -sashcursor sb_h_double_arrow
    pack $topFrame.panes -fill both -expand 1
    $topFrame.panes add leftFrame -margin 0 -minimum 0
    $topFrame.panes add rightFrame -margin 0 -minimum 0
    set leftFrame [$topFrame.panes childsite leftFrame].pad
    frame $leftFrame -borderwidth 5 -relief flat -bg $Color(Basecolor)
    pack $leftFrame -side left -fill both -expand 1 
    set rightFrame [$topFrame.panes childsite rightFrame].pad
    frame $rightFrame -borderwidth 5 -relief flat -bg $Color(Basecolor)
    pack $rightFrame -side left -fill both -expand 1 


    frame $botFrame.frame -relief sunken -borderwidth 3 -bg $Color(Basecolor)
    canvas $maincanvas -bg "$Color(NetworkEditor)" \
        -scrollregion "0 0 $mainCanvasWidth $mainCanvasHeight"
    pack $maincanvas -expand 1 -fill both
	
    # bgRect is just a rectangle drawn on the neteditFrame Canvas
    # so that the Modules List Menu can be bound to it using mouse
    # button 3.  The Modules List Menu can't be bound to the canvas
    # itself because mouse events are sent to both the objects on the
    # canvas (such as the lines connection the modules) and the canvas.
     eval $maincanvas create rectangle [$maincanvas cget -scrollregion] \
 	-fill "$Color(NetworkEditor)" -outline "$Color(NetworkEditor)" \
 	-tags bgRect 

    $maincanvas configure \
	-xscrollcommand "updateViewAreaBox; $botFrame.hscroll set" \
	-yscrollcommand "updateViewAreaBox; $botFrame.vscroll set"

    scrollbar $botFrame.hscroll -relief sunken -orient horizontal \
	-command "$maincanvas xview"
    scrollbar $botFrame.vscroll -relief sunken \
	-command "$maincanvas yview"


    # Layout the scrollbars and canvas in the bottom pane
    grid $botFrame.frame $botFrame.vscroll $botFrame.hscroll
    grid columnconfigure $botFrame 0 -weight 1 
    grid rowconfigure    $botFrame 0 -weight 1 
    grid config $botFrame.frame -column 0 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "snew" 
    grid config $botFrame.hscroll -column 0 -row 1 \
	    -columnspan 1 -rowspan 1 -sticky "ew" -pady 2
    grid config $botFrame.vscroll -column 1 -row 0 \
	    -columnspan 1 -rowspan 1 -sticky "sn" -padx 0

    # Create Error Message Window...
    text $rightFrame.text -relief sunken -bd 3 \
	-bg "$Color(ErrorFrameBG)" -fg "$Color(ErrorFrameFG)" \
	-yscrollcommand "$rightFrame.s set";# -height 10 -width 180 
    $rightFrame.text insert end "SCIRun v[netedit getenv SCIRUN_VERSION]\n"
    $rightFrame.text insert end "Messages:\n"
    $rightFrame.text insert end "--------------------------\n\n"
    $rightFrame.text tag configure errtag -foreground red
    $rightFrame.text tag configure warntag -foreground orange
    $rightFrame.text tag configure infotag -foreground yellow
    scrollbar $rightFrame.s -relief sunken -command "$rightFrame.text yview"
    pack $rightFrame.s -side right -fill y -padx 0 -ipadx 0
    pack $rightFrame.text -expand yes -fill both

    # Create Mini Network Editor
    global miniCanvasWidth miniCanvasHeight
    frame $leftFrame.frame -relief sunken -borderwidth 3 -bg "$Color(Basecolor)"
    canvas $minicanvas -bg $Color(NetworkEditor) 
    pack $minicanvas -expand 1 -fill both
    pack $leftFrame.frame -expand 1 -fill both
    $minicanvas create rectangle 0 0 1 1 -outline black -tag "viewAreaBox"
    initInfo

    .topbot fraction 25 75
    $topFrame.panes fraction 25 75
    wm withdraw .    
}

proc canvasScroll { canvas { dx 0.0 } { dy 0.0 } } {
    if {$dx!=0.0} {$canvas xview moveto [expr $dx+[lindex [$canvas xview] 0]]}
    if {$dy!=0.0} {$canvas yview moveto [expr $dy+[lindex [$canvas yview] 0]]}
}

# Activate the "File" menu items - called from C after all packages are loaded
proc activate_file_submenus { } {
    global maincanvas minicanvas    
    
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
    redrawMinicanvas
    bind $minicanvas <B1-Motion> "updateCanvases %x %y"
    bind $minicanvas <1> "updateCanvases %x %y"
    bind $minicanvas <Configure> "redrawMinicanvas"
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
    bind . <Control-d> "moduleDestroySelected"
    bind . <Control-l> "ClearCanvas"
    bind . <Control-z> "undo"
    bind . <Control-a> "selectAll"
    bind . <Control-e> "netedit scheduleall"
    bind . <Control-y> "redo"
    bind . <Control-o> "popupLoadMenu"
    bind . <Control-s> "popupSaveMenu"
    bind all <Control-q> "NiceQuit"
}

proc modulesMenu { subnet x y } {
    global mouseX mouseY Subnet
    set mouseX $x
    set mouseY $y
    set canvas $Subnet(Subnet${subnet}_canvas)
    createModulesMenu $canvas.modulesMenu $subnet
    tk_popup $canvas.modulesMenu [expr $x + [winfo rootx $canvas]] \
	[expr $y + [winfo rooty $canvas]]
}

proc redrawMinicanvas {} {
    global SCALEX SCALEY minicanvas maincanvas mainCanvasWidth mainCanvasHeight
    set w [expr [winfo width $minicanvas]-2]
    set h [expr [winfo height $minicanvas]-2]
    set w [expr ($w<=0)?1:$w]
    set h [expr ($h<=0)?1:$h]
    set SCALEX [expr $mainCanvasWidth/$w]
    set SCALEY [expr $mainCanvasHeight/$h]
    updateViewAreaBox
    global Subnet
    set connections ""
    $minicanvas raise module
    foreach module $Subnet(Subnet0_Modules) {
	set coords [scalePath [$maincanvas bbox $module]]
	after 1 $minicanvas coords $module $coords	
	eval lappend connections $Subnet(${module}_connections)
    }
    after 1 drawConnections \{[lsort -unique $connections]\}
}

proc updateViewAreaBox {} {
    global minicanvas maincanvas
    set w [expr [winfo width $minicanvas]-2]
    set h [expr [winfo height $minicanvas]-2]
    set ulx [expr [lindex [$maincanvas xview] 0] * $w]
    set lrx [expr [lindex [$maincanvas xview] 1] * $w]
    set uly [expr [lindex [$maincanvas yview] 0] * $h]
    set lry [expr [lindex [$maincanvas yview] 1] * $h]
    $minicanvas coords viewAreaBox $ulx $uly $lrx $lry
}

proc updateCanvases { x y } {
    global miniCanvasWidth miniCanvasHeight maincanvas minicanvas
    set x [expr $x/([winfo width $minicanvas]-2.0)]
    set y [expr $y/([winfo height $minicanvas]-2.0)]
    set xview [$maincanvas xview]
    set yview [$maincanvas yview]
    set x [expr $x-([lindex $xview 1]-[lindex $xview 0])/2]
    set y [expr $y-([lindex $yview 1]-[lindex $yview 0])/2]
    $maincanvas xview moveto $x
    $maincanvas yview moveto $y
    updateViewAreaBox
}

proc createPackageMenu {index} {
    global ModuleMenu ModuleIPorts ModuleOPorts    
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
	    set "ModuleIPorts(${package} ${category} ${module})" \
		[netedit module_iport_datatypes $package $category $module]
	    set "ModuleOPorts(${package} ${category} ${module})" \
		[netedit module_oport_datatypes $package $category $module]
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
    global maincanvas
    update idletasks
}


# createModulesMenu is called when the user right-clicks on the
# canvas.  It presents them with a menu of all modules.  Selecting
# a module name will create it at the clicked location
proc createModulesMenu { menu subnet } {
    global ModuleMenu Subnet
    # return if there is no information to put in menu
    if ![info exists ModuleMenu] return
    # destroy the old menu
#    if [winfo exists $menu] {	
#	destroy $menu
#    }
    # create a new menu
    if { ![winfo exists $menu] } {	
	menu $menu
    }
    $menu delete 0 end
    $menu configure -tearoff false -disabledforeground black

    foreach pack $ModuleMenu(packages) {
	# Add a menu separator if this package isn't the first one
	if { [$menu index end] != "none" } {
	    $menu add separator 
	}
	# Add a label for the Package name
	$menu add command -label "$ModuleMenu($pack)" -state disabled
	foreach cat $ModuleMenu(${pack}_categories) {
	    # Add the category to the right-button menu
	    $menu add cascade -label "  $ModuleMenu($cat)" -menu $menu.$cat
	    if { ![winfo exists $menu.$cat] } {	
		menu $menu.$cat -tearoff false
	    }
	    $menu.$cat delete 0 end

	    foreach mod $ModuleMenu(${pack}_${cat}_modules) {
		$menu.$cat add command -label "$ModuleMenu($mod)" \
		    -command "addModuleAtMouse \"$ModuleMenu($pack)\" \"$ModuleMenu($cat)\" \"$ModuleMenu($mod)\" \"$subnet\""
	    }
	}
    }
    
    $menu add separator
    $menu add cascade -label "Sub-Networks" -menu $menu.subnet
    if { ![winfo exists $menu.subnet] } {	
	menu $menu.subnet -tearoff false
    }

    createSubnetMenu $menu $subnet
	
    update idletasks
}

proc createSubnetMenu { { menu "" } { subnet 0 } } {
    global SubnetScripts
    loadSubnetScriptsFromDisk
    #generateSubnetScriptsFromNetwork

    if { [winfo exists $menu ] } {
	$menu.subnet delete 0 end
    }
    .main_menu.subnet.menu delete 0 end
    set names [lsort -dictionary [array names SubnetScripts *]]

    if { ![llength $names] } {
	if { [winfo exists $menu ] } {
	    $menu entryconfigure Sub-Networks -state disabled
	}
	.main_menu configure Sub-Networks configure -state disabled
    } else {
	if { [winfo exists $menu ] } {
	    $menu entryconfigure Sub-Networks -state normal
	}
	.main_menu.subnet configure -state normal
	foreach name $names {
	    if { [winfo exists $menu ] } {
		$menu.subnet add command -label "$name" \
		    -command "instanceSubnet \"$name\" 0 0 $subnet"
	    }
	    .main_menu.subnet.menu add command -label "$name" \
		-command "instanceSubnet \"$name\" 0 0 $subnet"
	}
    }
}
    

proc networkHasChanged {args} {
    upvar \#0 NetworkChanged changed
#    puts "$changed networkHasChanged [info level [expr [info level]-1]]"
    set changed 1
}

proc addModule { package category module } {
    return [addModuleAtPosition "$package" "$category" "$module" 10 10]
}

proc addModuleAtMouse { pack cat mod subnet_id } {
    global mouseX mouseY Subnet
    set Subnet(Loading) $subnet_id
    set ret [addModuleAtPosition $pack $cat $mod $mouseX $mouseY]
    set Subnet(Loading) 0
    set mouseX 10
    set mouseY 10
    return $ret
}


proc findMovedModulePath { packvar catvar modvar } {
    # Deprecated module translation table.
    set xlat "
{Fusion Fields NrrdFieldConverter} {Teem Converters NrrdToField}
{SCIRun FieldsCreate GatherPoints} {SCIRun FieldsCreate GatherFields}
{SCIRun Fields GatherPoints} {SCIRun FieldsCreate GatherFields}
{Teem DataIO ColorMapToNrrd} {Teem Converters ColorMapToNrrd}
{Teem DataIO FieldToNrrd} {Teem Converters FieldToNrrd}
{Teem DataIO NrrdToMatrix} {Teem Converters NrrdToMatrix}
{Teem DataIO MatrixToNrrd} {Teem Converters MatrixToNrrd}
{Teem DataIO NrrdToField} {Teem Converters NrrdToField}
{SCIRun Visualization NrrdToColorMap2} {Teem Converters NrrdToColorMap2}
{SCIRun Visualization GLTextureBuilder} {SCIRun Visualization TextureBuilder}
{SCIRun Visualization TextureVolVis} {SCIRun Visualization VolumeVisualizer}
{SCIRun Visualization TexCuttingPlanes} {SCIRun Visualization VolumeSlicer}
{SCIRun FieldsData ChangeFieldDataAt} {SCIRun FieldsData ChangeFieldBasis}
{SCIRun Fields ChangeFieldDataAt} {SCIRun FieldsData ChangeFieldBasis}
{SCIRun Visualization GenTransferFunc} {SCIRun Visualization EditColorMap}
{SCIRun Visualization EditTransferFunc2} {SCIRun Visualization EditColorMap2D}
{SCIRun FieldsData BuildInterpMatrix} {SCIRun FieldsData BuildMappingMatrix}
{SCIRun FieldsData ApplyInterpMatrix} {SCIRun FieldsData ApplyMappingMatrix}
{SCIRun FieldsData DirectInterpolate} {SCIRun FieldsData DirectMapping}
{SCIRun Fields DirectInterpolate} {SCIRun FieldsData DirectMapping}
{SCIRun FieldsData BuildInterpolant} {SCIRun FieldsData BuildMappingMatrix}
{SCIRun FieldsData ApplyInterpolant} {SCIRun FieldsData ApplyMappingMatrix}
{SCIRun Fields BuildInterpolant} {SCIRun FieldsData BuildMappingMatrix}
{SCIRun Fields ApplyInterpolant} {SCIRun FieldsData ApplyMappingMatrix}
"

    upvar 1 $packvar package $catvar category $modvar module
    set newpath [string map $xlat "$package $category $module"]
    set package  [lindex $newpath 0]
    set category [lindex $newpath 1]
    set module   [lindex $newpath 2]
}	        


proc addModuleAtPosition {package category module { xpos 10 } { ypos 10 } { absolute 0 } { modid "" } } {
    # Look up the real category for a module.  This allows networks to
    # be read in if the modules change categories.
    findMovedModulePath package category module
    set category [netedit getCategoryName $package $category $module]

    # fix for bug #2052, allows addmodule to call undefined procs without error
    set unknown_body [info body unknown]
    proc unknown { args } {}
    
    # default argument is empty, but if C already created the module, 
    # it will pass the id in.
    if { $modid == "" } {
	# Tell the C++ network to create the requested module
	set modid [netedit addmodule "$package" "$category" "$module"]
    }
    # Reset the unknown proc to default behavior
    proc unknown { args } $unknown_body

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
    set absolute [expr $absolute || $inserting]
    $modid make_icon $xpos $ypos $absolute
    update idletasks
    return $modid
}


proc addModuleAtAbsolutePosition {package category module { xpos 10 } { ypos 10 } { modid "" } } {
    # Look up the real category for a module.  This allows networks to
    # be read in if the modules change categories.
    findMovedModulePath package category module
    set category [netedit getCategoryName $package $category $module]

    # fix for bug #2052, allows addmodule to call undefined procs without error
    set unknown_body [info body unknown]
    proc unknown { args } {}
    
    # default argument is empty, but if C already created the module, 
    # it will pass the id in.
    if { $modid == "" } {
	# Tell the C++ network to create the requested module
	set modid [netedit addmodule "$package" "$category" "$module"]
    }
    # Reset the unknown proc to default behavior
    proc unknown { args } $unknown_body

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
    }
    $modid make_icon $xpos $ypos 1
    update idletasks
    return $modid
}



# addModule2 creates a SCIRun module to be used in the SCIRun2 framework
# as an instance of the SCIRunComponentModel.
proc addModule2 {package category module modid} {  
    global Subnet

    set Subnet($modid) $Subnet(Loading)
    set Subnet(${modid}_connections) ""
    lappend Subnet(Subnet$Subnet(Loading)_Modules) $modid

    set className [join "${package}_${category}_${module}" ""]
    if {[catch "$className $modid" exception]} {
        # Use generic module
        if {$exception != "invalid command name \"$className\""} {
            bgerror "Error instantiating iTcl class for module:\n$exception";
        }
        Module $modid -name "$module"
    }

    redrawMinicanvas
    $modid make_icon 10 10 0

    return $modid
}

proc append_srn_filename {name} {

    set ext_ind [expr [string length $name] - 4]
    set ext [string range $name $ext_ind end]
    
    if { $ext == ".net" } {
	set name [string range $name 0 $ext_ind]srn
	createSciDialog -warning -title "Save Warning" -button1 "Ok"\
	    -message "SCIRun no longer saves .net files.\nSaving $name instead."
	set ext ".srn"
    } 
    
    if { $ext != ".srn" } {
	set name $name.srn
    } 
    return $name
}


proc popupSaveMenu {} {
    global netedit_savefile NetworkChanged
    if { $netedit_savefile != "" } {
	# We know the name of the savefile, dont ask for name, just save it
	# make sure we only save .srn files
	set netedit_savefile [append_srn_filename $netedit_savefile]
	wm title . "SCIRun ([lindex [file split $netedit_savefile] end])"
	writeNetwork $netedit_savefile
	set NetworkChanged 0
    } else { ;# Otherwise, ask the user for the name to save as
	popupSaveAsMenu
    }
}

proc popupSaveAsMenu {} {
    set types {
	{{SCIRun Net} {.srn} }
    } 

    global netedit_savefile NetworkChanged

    # determine initialdir based on current $netedit_savefile
    set dirs [file split "$netedit_savefile"]
    set initialdir [pwd]
	set initialfile ""

    if {[llength $dirs] > 1} {
	set initialdir ""
	set size [expr [llength $dirs] - 1]
	for {set i 0} {$i<$size} {incr i} {
	    set initialdir [file join $initialdir [lindex $dirs $i]]
	}
	set initialfile [lindex $dirs $size]
    }

    set netedit_savefile \
	[tk_getSaveFile -defaultextension {.srn} -filetypes $types -initialdir $initialdir -initialfile $initialfile]
    if { $netedit_savefile != "" } {
	# make sure we only save .srn files
	set netedit_savefile [append_srn_filename $netedit_savefile]
	wm title . "SCIRun ([lindex [file split $netedit_savefile] end])"
	writeNetwork $netedit_savefile
	set NetworkChanged 0
    }
}

proc popupInsertMenu { {subnet 0} } {
    global inserting insertOffset Subnet NetworkChanged
    global mainCanvasWidth mainCanvasHeight
    
    #get the net to be inserted
    set types {
      {{SCIRun Net} {.srn} }
      {{old SCIRun Net} {.net} }
    } 
    set netedit_insertnet [tk_getOpenFile -filetypes $types ]
    if { [check_filename $netedit_insertnet] == "invalid" } {
      set netedit_insertnet ""
      return
    }
    if { ![file exists $netedit_insertnet]} { 
      return
    }
    
    set canvas $Subnet(Subnet${subnet}_canvas)    
    # get the bbox for the net being inserted by
    # parsing netedit_loadnet for bbox 
    set fchannel [open $netedit_insertnet]
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
    if {[string match *.net $netedit_insertnet]} {
      loadnet $netedit_insertnet
    } else {
      global netedit_savefile
      set tmp $netedit_savefile
      uplevel \#0 netedit load_srn $netedit_insertnet
      set netedit_savefile $tmp
    }
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


proc check_filename {name} {

	if {$name == ""} {
		return "invalid"
	}

    set ext_ind [expr [string length $name] - 4]
    set ext [string range $name $ext_ind end]
    
    if { $ext != ".net" && $ext != ".srn"} {
		set name [string range $name 0 $ext_ind]srn
		set msg "Valid net files end with .srn (or .net prior to v1.25.2)"
		createSciDialog -warning -title "Save Warning" -button1 "Ok"\
			-message $msg
		return "invalid"
    } 
    return "valid"
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
	{{SCIRun Net} {.srn} }
	{{old SCIRun Net} {.net} }
    } 
    
    set netedit_loadnet [tk_getOpenFile -filetypes $types ]
    if { [check_filename $netedit_loadnet] == "invalid" } {
		set netedit_loadnet ""
		return
    }

    if { ![file exists $netedit_loadnet]} { return }

    #dont ask user before clearing canvas
    ClearCanvas 0
    set inserting 0
    if {[string match *.srn $netedit_loadnet]} {
        # compensate for spaces in the filename (windows)
		after 500 uplevel \#0 netedit load_srn \{\{$netedit_loadnet\}\}
    } else {
		loadnet $netedit_loadnet 
    }
}

proc ClearCanvas { { confirm 1 } { subnet 0 } } {
    # destroy all modules
    global NetworkChanged
    if { !$NetworkChanged } { set confirm 0 }
    set do_clear yes
    if { $confirm } {
	set message [list "Your network has not been saved." \
			 "All Modules and connections will be deleted." \
			 "Really clear?"]
	set do_clear [tk_messageBox -title "Warning" -type yesno -parent . \
			  -icon warning -message [join $message "\n\n"] ]
    }

    if { $do_clear == "yes" } {
	global Subnet
	foreach module $Subnet(Subnet${subnet}_Modules) {
	    if { [string first Render_Viewer $module] != -1 } {
		moduleDestroy $module
	    }
	}
	foreach module $Subnet(Subnet${subnet}_Modules) {
	    moduleDestroy $module
	}

	wm title . "SCIRun" ;# Reset Main Window Title
	setGlobal netedit_savefile ""
	setGlobal CurrentlySelectedModules ""
	setGlobal NetworkChanged 0
    }
}

proc NiceQuit {} {
    global NetworkChanged netedit_savefile
    if { $NetworkChanged && ![envBool SCIRUN_FAST_QUIT] } {
	set result [createSciDialog -warning -title "Quit?" \
		    -button1 "Don't Save" -button2 "Cancel" -button3 "Save" \
		    -message "Your session has not been saved.\nWould you like to save before exiting?"]
	switch -- $result { 
	    "-1" return
	    "2" return
	    "3" {
		if { [winfo exists .standalone] } {
		    app save_session
		} else {
		    puts -nonewline "Saving $netedit_savefile..."
		    popupSaveMenu
		}
	    }
	}
    }
    set geom [open ~/.scirun.geom w]
    puts $geom [wm geom .]
    close $geom
    puts "Goodbye!"
    netedit quit
}

proc initInfo { {subnet_number 0} } {
    global Subnet
    if { ![info exists Subnet(Subnet${subnet_number}_userName)] } {
	set Subnet(Subnet${subnet_number}_userName) [netedit getenv LOGNAME] 
	if { $Subnet(Subnet${subnet_number}_userName) == "" } {
	    set Subnet(Subnet${subnet_number}_userName) [netedit getenv USER] 
	}
	if { $Subnet(Subnet${subnet_number}_userName) == "" } {
	    set Subnet(Subnet${subnet_number}_userName) Unknown
	}
    }
    if { ![info exists Subnet(Subnet${subnet_number}_creationDate)] } {
	set Subnet(Subnet${subnet_number}_creationDate) \
	    [clock format [clock seconds] -format "%a %b %d %Y"]
    }
    if { ![info exists Subnet(Subnet${subnet_number}_creationTime)] } {
	set Subnet(Subnet${subnet_number}_creationTime) \
	    [clock format [clock seconds] -format "%H:%M:%S"]
    }
    if { ![info exists Subnet(Subnet${subnet_number}_runDate)] } {
	set Subnet(Subnet${subnet_number}_runDate) ""
    }
    if { ![info exists Subnet(Subnet${subnet_number}_runTime)] } {
	set Subnet(Subnet${subnet_number}_runTime) ""
    }

    if { ![info exists Subnet(Subnet${subnet_number}_notes)] } { 
	set Subnet(Subnet${subnet_number}_notes) "" 
    }
		
		if { ![info exists Subnet(Subnet${subnet_number}_relfilenames)]} {
			set Subnet(Subnet${subnet_number}_relfilenames) 0
		}
}

proc updateRunDateAndTime { subnet_number } {
    global Subnet
    set Subnet(Subnet${subnet_number}_runDate) \
	[clock format [clock seconds] -format "%a %b %d %Y"]
    set Subnet(Subnet${subnet_number}_runTime) \
	[clock format [clock seconds] -format "%H:%M:%S"]
}    


proc updateCreationDateAndTime { subnet_number } {
    global Subnet
    set Subnet(Subnet${subnet_number}_creationDate) \
	[clock format [clock seconds] -format "%a %b %d %Y"]
    set Subnet(Subnet${subnet_number}_creationTime) \
	[clock format [clock seconds] -format "%H:%M:%S"]
}    

# This proc was added by Mohamed Dekhil to save some info about the net
proc popupInfoMenu { {subnet_num 0 } } {
    global Subnet 
    initInfo $subnet_num
    
    lappend backupDateTimeNotes \
	$Subnet(Subnet${subnet_num}_userName) \
	$Subnet(Subnet${subnet_num}_runDate) \
	$Subnet(Subnet${subnet_num}_runTime) \
	$Subnet(Subnet${subnet_num}_creationDate) \
	$Subnet(Subnet${subnet_num}_creationTime) \
	$Subnet(Subnet${subnet_num}_notes) \
	$Subnet(Subnet${subnet_num}_relfilenames) \

    set w .netedit_info$subnet_num
        
    if {[winfo exists $w]} {
			raise $w
			return
    }

    toplevel $w
    wm title $w "$Subnet(Subnet${subnet_num}_Name) Properties"

		iwidgets::tabnotebook $w.tab -tabpos n  -width 600 -height 600
		
		pack $w.tab -fill both -expand yes
		set infoframe [$w.tab add -label "Information" ]
		# set saveframe [$w.tab add -label "Save Options" ] 
		
		$w.tab select 0

		# disable for now until we resolve problems with srn files
		# frame $saveframe.fname
		# checkbutton $saveframe.fname.relativefiles -variable Subnet(Subnet${subnet_num}_relfilenames) -text "Save all filenames relative to the network location"
		
		# pack $saveframe.fname -side top -padx 1 -pady 1 -ipadx 2 -ipady 2 -fill x
		# pack $saveframe.fname.relativefiles -side left

    frame $infoframe.fname
    label $infoframe.fname.lname -text "User: " -padx 3 -pady 3
    entry $infoframe.fname.ename -width 50 -relief sunken -bd 2 \
	-textvariable Subnet(Subnet${subnet_num}_userName)
    pack $infoframe.fname.lname $infoframe.fname.ename -side left

    set pre [expr $subnet_num?"Sub-":""]
    frame $infoframe.cdt
    label $infoframe.cdt.label -text "${pre}Network Created:"
    label $infoframe.cdt.ldate -text "   Date: " -padx 3 -pady 3 
    entry $infoframe.cdt.edate -width 20 -relief sunken -bd 2 \
	-textvariable Subnet(Subnet${subnet_num}_creationDate)

    label $infoframe.cdt.ltime -text "   Time: " -padx 5 -pady 3 
    entry $infoframe.cdt.etime -width 10 -relief sunken -bd 2 \
	-textvariable Subnet(Subnet${subnet_num}_creationTime)

    button $infoframe.cdt.reset -text "Reset" \
	-command "updateCreationDateAndTime $subnet_num"
    pack $infoframe.cdt.label  -side left -fill x
    pack $infoframe.cdt.reset $infoframe.cdt.etime $infoframe.cdt.ltime $infoframe.cdt.edate $infoframe.cdt.ldate   -side right -padx 5


    frame $infoframe.fdt
    label $infoframe.fdt.label -text "${pre}Network Executed:"
    label $infoframe.fdt.ldate -text "   Date: " -padx 3 -pady 3 
    entry $infoframe.fdt.edate -width 20 -relief sunken -bd 2 \
	-textvariable Subnet(Subnet${subnet_num}_runDate)

    label $infoframe.fdt.ltime -text "   Time: " -padx 5 -pady 3 
    entry $infoframe.fdt.etime -width 10 -relief sunken -bd 2 \
	-textvariable Subnet(Subnet${subnet_num}_runTime)

    button $infoframe.fdt.reset -text "Reset" \
	-command "updateRunDateAndTime $subnet_num"

    pack $infoframe.fdt.label  -side left -fill x
    pack $infoframe.fdt.reset $infoframe.fdt.etime $infoframe.fdt.ltime $infoframe.fdt.edate $infoframe.fdt.ldate   -side right -padx 5


    frame $infoframe.fnotes -relief groove
    frame $infoframe.fnotes.top
    frame $infoframe.fnotes.bot
    label $infoframe.fnotes.top.lnotes -text "Notes:" -padx 2 -pady 5 
    text $infoframe.fnotes.bot.tnotes -relief sunken -bd 2 \
	-yscrollcommand "$infoframe.fnotes.bot.scroll set"
    scrollbar $infoframe.fnotes.bot.scroll -command "$infoframe.fnotes.bot tnotes yview"
    $infoframe.fnotes.bot.tnotes insert 1.0 $Subnet(Subnet${subnet_num}_notes)

    pack $infoframe.fnotes.top $infoframe.fnotes.bot -expand 0 -fill x -side top
    pack $infoframe.fnotes.top.lnotes -side left
    pack $infoframe.fnotes.bot -expand 1 -fill both -side top
    pack $infoframe.fnotes.bot.tnotes  -expand 1 -side left -fill both
    pack $infoframe.fnotes.bot.scroll -expand 0 -side left -fill y


    frame $w.fbuttons 
    button $w.fbuttons.ok -text "Done" -command "infoOk $w $subnet_num"
    button $w.fbuttons.clear -text "Clear All" \
	-command "infoClear $w $subnet_num"
    button $w.fbuttons.cancel -text "Cancel" \
	-command "infoCancel $w $subnet_num $backupDateTimeNotes"

    pack $infoframe.fname $infoframe.cdt $infoframe.fdt -side top -padx 1 -pady 1 -ipadx 2 -ipady 2 -fill x
    pack $infoframe.fnotes -expand 1 -fill both

    pack $w.fbuttons -side top -padx 1 -pady 1 -ipadx 2 -ipady 2 -fill x
    pack $w.fbuttons.ok $w.fbuttons.clear $w.fbuttons.cancel -side right -padx 5 -pady 5 -ipadx 3 -ipady 3
}

proc infoClear {w subnet_num} {
    global Subnet 
    set Subnet(Subnet${subnet_num}_userName) ""
    set Subnet(Subnet${subnet_num}_runDate) ""
    set Subnet(Subnet${subnet_num}_runTime) ""
    set Subnet(Subnet${subnet_num}_creationDate) ""
    set Subnet(Subnet${subnet_num}_creationTime) ""
    set Subnet(Subnet${subnet_num}_notes) ""
		set Subnet(Subnet${subnet_num}_relfilenames) ""
		
		set infoframe [$w.tab childsite 0]		
    $infoframe.fnotes.bot.tnotes delete 1.0 end
}

proc infoOk {w subnet_num} {
    global Subnet
		set infoframe [$w.tab childsite 0]
    set Subnet(Subnet${subnet_num}_notes) [$infoframe.fnotes.bot.tnotes get 1.0 end]
    networkHasChanged
    destroy $w
}

proc infoCancel {w subnet_num args } {
    global Subnet
    set Subnet(Subnet${subnet_num}_userName) [lindex $args 0]
    set Subnet(Subnet${subnet_num}_runDate) [lindex $args 1]
    set Subnet(Subnet${subnet_num}_runTime) [lindex $args 2]
    set Subnet(Subnet${subnet_num}_creationDate) [lindex $args 3]
    set Subnet(Subnet${subnet_num}_creationTime) [lindex $args 4]
    set Subnet(Subnet${subnet_num}_notes) [lindex $args 5]
		set Subnet(Subnet${subnet_num}_relfilenames) [lindex $args 6]

    destroy $w
} 

proc loadfile {netedit_loadfile} {
    puts "NOTICE: `loadfile' has been disabled."
    puts "   To use old nets, remove the `loadfile' and `return' lines"
    puts "   from near the top of the file."
    return
}

proc loadnet { netedit_loadfile} {
    # Check to see of the file exists; warn user if it doesnt
    if { ![file exists $netedit_loadfile] } {
	set message "File \"$netedit_loadfile\" does not exist."
	createSciDialog -warning -message $message
	return
    }

    global netedit_savefile inserting PowerApp Subnet geometry
    if { !$inserting || ![string length $netedit_savefile] } {
      # Cut off the path from the net name and put in on the title bar:
      wm title . "SCIRun ([lindex [file split $netedit_loadfile] end])"
      # Remember the name of this net for future "Saves".
      set netedit_savefile $netedit_loadfile
    }

    renameSourceCommand
    
    # if we are not loading a powerapp network, show loading progress
    if { !$PowerApp } {
	showProgress 0 0 1 ;# -maybe- raise the progress meter
	setProgressText "Loading SCIRun network file..."
	update idletasks
	# The following counts the number of steps it will take to load the net
	resetScriptCount
	renameNetworkCommands counting_
	source $netedit_loadfile
	renameNetworkCommands counting_
	global scriptCount
	addProgressSteps $scriptCount(Total)
    }

    # Renames a few procs to increment the progressmeter as we load
    renameNetworkCommands loading_

    uplevel \#0 source \{$netedit_loadfile\}
    
    # Restores the original network loading procedures
    renameNetworkCommands loading_
    resetSourceCommand

    if { !$PowerApp } {
	hideProgress
    }
    set Subnet(Subnet$Subnet(Loading)_Filename) $netedit_loadfile
    if { !$inserting } { setGlobal NetworkChanged 0 }
}

proc SCIRunNew_source { args } {
    set filename [lindex $args 0]
    if { [file exists $filename] } {    
	return [uplevel 1 SCIRunBackup_source \{$filename\}]
    }

    set lastSettings [string last .settings $filename]
    if { ($lastSettings != -1) && \
	     ([expr [string length $filename] - $lastSettings] == 9) } {
	set file "[netedit getenv SCIRUN_SRCDIR]/nets/default.settings"
	global recentlyWarnedAboutDefaultSettings
	if { ![info exists recentlyWarnedAboutDefaultSettings] } {
	    set recentlyWarnedAboutDefaultSettings 1
	    after 10000 uplevel \#0 unset recentlyWarnedAboutDefaultSettings
	    displayErrorWarningOrInfo "*** No $filename file was found.\n*** Loading the default dataset .settings file: $file\n" info
	}
	return [uplevel 1 SCIRunBackup_source \{$file\}]
    }
    puts "SCIRun TCL cannot source \'$filename\': File does not exist."
}

proc renameSourceCommand {} {
    if { [llength [info commands SCIRunBackup_source]] } return
    rename source SCIRunBackup_source
    rename SCIRunNew_source source
}



proc resetSourceCommand {} {
    if { ![llength [info commands SCIRunBackup_source]] } return
    rename source SCIRunNew_source
    rename SCIRunBackup_source source

}

proc showInvalidDatasetPrompt {} {
    set filename [lindex [file split [info script]] end]
    return [createSciDialog -warning \
		-parent . \
		-button1 "Select Dataset" \
		-button2 "Ignore Dataset" \
		-button3 "Exit SCIRun" \
		-message "This network can work on different datasets, but it\ncannot find files matching the pattern:\n\$(SCIRUN_DATA)/\$(SCIRUN_DATASET)/\$(SCIRUN_DATAFILE)*\n\nWhere:\nSCIRUN_DATA = [netedit getenv SCIRUN_DATA]\nSCIRUN_DATASET = [netedit getenv SCIRUN_DATASET]\nSCIRUN_DATAFILE = [netedit getenv SCIRUN_DATAFILE]\n\nThe environment variables listed are not set or they are invalid.\n\nChoose 'Select Dataset' to select a valid dataset directory.\neg: /usr/sci/data/SCIRunData/[netedit getenv SCIRUN_VERSION]/sphere\n\nChoose 'Ignore Dataset' to load $filename anyway.  You will\nhave to manually set the reader modules to valid filenames.\n\nIf you set these environment variables before running SCIRun, then\nyou will not receive this dialog when loading $filename.\n\nYou may also permanently set these variables in the ~/.scirunrc file."]
}



proc showChooseDatasetPrompt { initialdir } {
    if { ![string length $initialdir] } {
	set version [netedit getenv SCIRUN_VERSION]
	set initialdir "/usr/sci/data/SCIRunData/${version}"
	if { ![file exists $initialdir] } {
	    set initialdir [netedit getenv SCIRUN_OBJDIR]
	}
    }
    set value [tk_chooseDirectory -mustexist 1 -initialdir $initialdir \
		  -parent . -title "Select Dataset"]
    return $value
}


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
    # Attempt to get environment variables:
    set DATADIR [netedit getenv SCIRUN_DATA]
    set DATASET [netedit getenv SCIRUN_DATASET]
    set DATAFILE [netedit getenv SCIRUN_DATAFILE]
    
    if { ![string length $DATASET] } {
	# if env var SCIRUN_DATASET not set... default to sphere:
	set DATASET sphere
    } 

    global recentlyCalledSourceSettingsFile
    if { ![info exists recentlyCalledSourceSettingsFile] } {
	set initialdir ""
	while {![string length [glob -nocomplain "$DATADIR/$DATASET/$DATAFILE*"]] } {
	    case [showInvalidDatasetPrompt] {
		1 "set data [showChooseDatasetPrompt $initialdir]"
		2 { 
		    displayErrorWarningOrInfo "*** SCIRUN_DATA not set.  Reader modules will need to be manually set to valid filenames." warning
		    uplevel \#0 source "[netedit getenv SCIRUN_SRCDIR]/nets/default.settings"
		    return
		}
		3 "netedit quit"
	    }
	    if { [string length $data] } {
		set initialdir $data
		set data [file split $data]
		set DATASET [lindex $data end]
		set DATADIR [eval file join [lrange $data 0 end-1]]
	    }	      
	}

	displayErrorWarningOrInfo "*** Using SCIRUN_DATA=$DATADIR" info
	displayErrorWarningOrInfo "*** Using SCIRUN_DATASET=$DATASET" info
	displayErrorWarningOrInfo "*** Using SCIRUN_DATAFILE=$DATAFILE" info
	
	netedit setenv SCIRUN_DATA "$DATADIR"
	netedit setenv SCIRUN_DATASET "$DATASET"
	netedit setenv SCIRUN_DATAFILE "$DATAFILE"

	setGlobal recentlyCalledSourceSettingsFile 1
	after 10000 uplevel \#0 unset recentlyCalledSourceSettingsFile
    }

    set settings "$DATADIR/$DATASET/$DATASET.settings"
    uplevel 1 source $settings
    return "$DATADIR $DATASET $DATAFILE"
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
    global rightFrame
    $rightFrame.text insert end "$msg\n" "$status_tag"
    $rightFrame.text see end
}


proc hideProgress { args } {
    if { ![winfo exists .splash] } return
    update idletasks
    update
    wm withdraw .splash
    grab release .splash
    update idletasks
}

proc showProgress { { show_image 0 } { steps none } { okbutton 0 } { over . } } {
    if { [envBool SCIRUN_HIDE_PROGRESS] && ![winfo exists .standalone] } return
    update idletasks
    set w .splash
    if { ![winfo exists $w] } {
	toplevel $w -bd 2 -relief raised
	wm withdraw $w
	wm protocol $w WM_DELETE_WINDOW hideProgress
    }
    if { $show_image} {
	wm title $w "Welcome to SCIRun v[netedit getenv SCIRUN_VERSION]"
    } else {
	wm title $w "Loading Network..."
    }

    if { ![winfo exists $w.frame] } {
	frame $w.frame
	pack $w.frame -expand 1 -fill both
    }
    set w $w.frame

    # do not show if either env vars are set to true, the only 
    # exception is when we are calling this from a powerapp's
    # show_help
    if { [envBool SCIRUN_NOSPLASH] || [envBool SCI_NOSPLASH] } {
	set show_image 0
    }
    if {[winfo exists .standalone]} {
	set show_image 1
    }

    if { [winfo exists $w.fb] } {
	destroy $w.fb
	unsetIfExists progressMeter
    }

    if { ![string equal $steps none] } {
	iwidgets::feedback $w.fb -steps 0
	setGlobal progressMeter $w.fb
    }

    if { ![winfo exists $w.ok] } {
	button $w.ok -text Dismiss -width 10 -command hideProgress
    }

    if { $show_image } {
        global splashImageFile
        if { [string length [info commands ::img::splash]] && \
                 ![string equal $splashImageFile [::img::splash cget -file]] } {
            image delete ::img::splash
        }

        if { [file isfile $splashImageFile] && \
                 [file readable $splashImageFile] && \
                 ![string length [info commands ::img::splash]] } {
            image create photo ::img::splash -file $splashImageFile
        }

        if { ![winfo exists $w.splash] } {
            label $w.splash
        }
    
        if { [string length [info command ::img::splash]] } {
            $w.splash configure -image ::img::splash
            pack $w.splash
        } else {
            pack forget $w.splash
        }
    } else {
	pack forget $w.splash
    }

    if { ![string equal $steps none ]} {
	setProgressText Initializing...
	addProgressSteps $steps
	pack $w.fb -padx 5 -fill x
    } else {
	pack forget $w.fb
    }

    if { $okbutton } {
	pack $w.ok -side bottom -padx 5 -pady 5 -fill none
    } else {
	pack forget $w.ok
    }

    if {![winfo exists $w.scaffold] } {
	frame $w.scaffold -width 500 -relief raised -bd 3
	pack $w.scaffold -side bottom
    }

    centerWindow .splash $over
}

proc addProgressSteps { steps } {
    global progressMeter
    if { [info exists progressMeter] && \
	     [winfo exists $progressMeter] } {
	set steps [expr $steps+[$progressMeter cget -steps]]	
	$progressMeter configure -steps [expr int(1.01*$steps)]
    }
}

proc setProgressText { text } {
    global progressMeter
    if { [info exists progressMeter] && \
	     [winfo exists $progressMeter] } {
	$progressMeter configure -labeltext $text
	update idletasks
    }
}

proc incrProgress { { steps 1 } } {
    global progressMeter
    if { ![info exists progressMeter] } return
    while { $steps } {
	$progressMeter step
	incr steps -1
    }
}

    
	 

global LicenseResult
set licenseResult decline


proc licenseDialog { {firsttime 0} } {
    if $firsttime { return "accept" }
    global licenseResult userData
    set filename [file join [netedit getenv SCIRUN_SRCDIR] LICENSE]
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
    global licenseResult userData
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
	$w.text insert end "The following e-mail was automatically generated by SCIrun v$[netedit getenv SCIRUN_VERSION]\n\n"
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
	    puts $out "\nThe following e-mail was automatically generated by SCIRun v$[neteedit getenv SCIRUN_VERSION]"
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

proc promptUserToCopySCIRunrc {} {
    global copyResult
    set w .copyRCprompt

    toplevel $w
    wm withdraw $w

    set copyResult 0
    set dontAskAgain 0

    set version [netedit getenv SCIRUN_RCFILE_VERSION]
    if { $version == "" } {
        set version "bak"
    }

    wm title $w "Copy v$version .scirunrc file?"
    label $w.message -text "A newer version of the .scirunrc file is avaliable with this release.\n\nThis file contains SCIRun environment variables that are\nneccesary for new features.\n\nPlease note: If you have made changes to your ~/.scirunrc file\nthey will be undone by copying.\n\nIf you copy, your existing file will be moved to\n ~/.scirunrc.$version\n\nWould you like SCIRun to copy over the new .scirunrc?\n" -justify left

    frame $w.but
    button $w.but.ok -text Copy -command "set copyResult 1"
    button $w.but.no -text "Don't Copy" -command "set copyResult 0"
    button $w.but.dontask -text "Don't Ask Again" -command "set copyResult 2"

    pack $w.but.ok $w.but.no $w.but.dontask  -side left -pady 5 -padx 5 -ipadx 5 -expand 1
    pack $w.message -expand 1 -fill both 
    pack $w.but -expand 1 -fill both -side top -anchor n

    # Override the destroy window decoration and make it not do anything
    wm protocol $w WM_DELETE_WINDOW "SciRaise $w"
    moveToCursor $w
    SciRaise $w

    vwait copyResult

    if { $copyResult == 2 } {
	if { [catch { set rcfile [open ~/.scirunrc "WRONLY APPEND"] }] } {
            puts "Unable to open ~/.scirunrc"
            return 0
        }

	puts $rcfile \
            "\n\# This section added when the user chose 'Dont Ask Again'"
	puts $rcfile \
            "\# when prompted about updating the .scirurc file version"

	set version [netedit getenv SCIRUN_VERSION]
        set rcversion [netedit getenv SCIRUN_RCFILE_SUBVERSION]
	puts $rcfile "SCIRUN_RCFILE_VERSION=${version}.${rcversion}"
	close $rcfile
        set copyResult 0
    }

    destroy $w
    return $copyResult
}
    
proc validFile { args } {
    set name [lindex $args 0]
    return [expr [file isfile $name] && [file readable $name]]
}

proc validDir { name } {
    return [expr [file isdirectory $name] && \
		 [file writable $name] && [file readable $name]]
}

proc getOnTheFlyLibsDir {} {
    global tcl_platform
    set binOTF [file join [netedit getenv SCIRUN_OBJDIR] on-the-fly-libs]

    set dir [netedit getenv SCIRUN_ON_THE_FLY_LIBS_DIR]

    if { $dir != "" } {
	catch "file mkdir $dir"
	if { [validDir $dir] && ![llength [glob -nocomplain -directory $dir *]] } {
	    displayErrorWarningOrInfo "\nCopying the contents of $binOTF\nto $dir..." info
	    update idletasks
	    foreach name [glob -nocomplain -directory $binOTF *.cc *.d *.o *.so] {
		file copy $name $dir
	    }
	    displayErrorWarningOrInfo "Done copying on-the-fly-libs.\n" info
	}
    }

    if ![validDir $dir] {
	set dir $binOTF
	
	# if this is a windows dir, it won't think this is a valid dir with the '/'s in the name
        set ostype [netedit getenv OS]
        if { [string equal $ostype "Windows_NT"] } {
            return $binOTF
        }
    }

    if ![validDir $dir] {
	set home [file nativename ~]
	set dir [file join $home SCIRun on-the-fly-libs $tcl_platform(os)]
	catch "file mkdir $dir"
	if { [validDir $dir] && ![llength [glob -nocomplain -directory $dir *]] } {
	    displayErrorWarningOrInfo "\nCopying the contents of $binOTF\nto $dir..." info
	    update idletasks
	    foreach name [glob -nocomplain -directory $binOTF *.cc *.d *.o *.so] {
		file copy $name $dir
	    }
	    displayErrorWarningOrInfo "Done copying on-the-fly-libs.\n" info
	}
    }
    if { ![validDir $dir] } {
	tk_messageBox -type ok -parent . -icon error -message \
	    "SCIRun cannot find a directory to store dynamically compiled code.\n\nPlease quit and set the environment variable SCIRUN_ON_THE_FLY_LIBS_DIR to a readable and writable directory.\n\nDynamic code generation will not work.  If you continue this session, networks may not execute correctly."
	return $binOTF
    }

    set makefile [file join [netedit getenv SCIRUN_OBJDIR] on-the-fly-libs Makefile]
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

# Finds the first instance of elem in a list then removes it from the list 
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


proc initVarStates { var save substitute } {
    set var [string trimleft $var :]
    if { [string first msg_stream $var] != -1 } return
    global ModuleSavedVars ModleSubstitutedVars
    set ids [split $var -]
    if { [llength $ids] < 2 } return
    set module [lindex $ids 0]
    set varname [join [lrange $ids 1 end] -]
    
    if { $save } {
	lappend ModuleSavedVars($module) $varname
	# TODO: find a faster mechanism than the next line for setting changed
	uplevel \#0 trace variable \"$var\" w networkHasChanged
    }
     
    if { $substitute } {
	lappend ModuleSubstitutedVars($module) $varname
    }
}

proc setVarStates { var save substitute isfilename} {

    global ModuleSavedVars ModuleSubstitutedVars ModuleIsFilenameVars
    set var [string trimleft $var :]
    if { [string first msg_stream $var] != -1 } return
    set ids [split $var -]
    set module [lindex $ids 0]
    set varname [join [lrange $ids 1 end] -]
    if { ![string length $varname] || ![string length $module] } return

    if { ![info exists ModuleSavedVars($module)] } {
			set saved 0
    } else {
			set saved [expr [lsearch $ModuleSavedVars($module) $varname] != -1]
    }

    if { $save && !$saved} {
			lappend ModuleSavedVars($module) $varname
			uplevel \#0 trace variable \"$var\" w networkHasChanged
    } elseif { !$save && $saved } {
			listFindAndRemove ModuleSavedVars($module) $varname
			uplevel \#0 trace vdelete \"$var\" w networkHasChanged
    }
	
    if { ![info exists ModuleSubstitutedVars($module)] } {
			set substituted 0
    } else {
			set substituted [expr [lsearch $ModuleSubstitutedVars($module) $varname] != -1]
    }

    if { $substitute && !$substituted } {
			lappend ModuleSubstitutedVars($module) $varname
    } elseif { !$substitute && $substituted } {
			listFindAndRemove ModuleSubstitutedVars($module) $varname
    }

    if { ![info exists ModuleIsFilenameVars($module)] } {
			set filename 0
    } else {
			set filename [expr [lsearch $ModuleIsFilenameVars($module) $varname] != -1]
    }

    if { $isfilename && !$filename } {
			lappend ModuleIsFilenameVars($module) $varname
    } elseif { !$isfilename && $filename } {
			listFindAndRemove ModuleIsFilenameVars($module) $varname
    }
}


# Debug procedure to print global variable values
proc printvars { pattern } {
    foreach name [lsort [uplevel \#0 "info vars *${pattern}*"]] { 
	upvar \#0 $name var
	if { [uplevel \#0 array exists \"$name\"] } {
	    uplevel \#0 parray \"$name\"
	} else {
	    puts "set \"$name\" \{$var\}"
	}
    }
}

proc setGlobal { var val } {
    uplevel \#0 set \"$var\" \{$val\}
}

# Will only set global variable $var to $val if it doesnt already exist
proc initGlobal { var val } {
    upvar \#0 $var globalvar
    if { ![info exists globalvar] } {
	set globalvar $val
    }
}


proc popFront { listname } {
    upvar 1 $listname list
    if ![info exists list] return
    set front [lindex $list 0]
    set list [lrange $list 1 end]
    return $front
}

proc popBack { listname } {
    upvar 1 $listname list
    if ![info exists list] return
    set back [lindex $list end]
    set list [lrange $list 0 end-1]
    return $back
}

proc maybeWrite_init_DATADIR_and_DATASET { out } {
    global ModuleSubstitutedVars
    if { ![envBool SCIRUN_NET_SUBSTITUTE_DATADIR] } return
    foreach module [array names ModuleSubstitutedVars] {
	foreach var $ModuleSubstitutedVars($module) {
	    upvar \#0 $module-$var val
	    if { [info exists val] && \
		![string equal $val [subDATADIRandDATASET $val]] } {
		netedit net-add-env-var scisub_datadir SCIRUN_DATA
		netedit net-add-env-var scisub_datafile SCIRUN_DATAFILE
		netedit net-add-env-var scisub_dataset SCIRUN_DATASET
		return
	    }
	}
    }
}


proc maybeWriteTCLStyleCopyright { out } {
    if { ![envBool SCIRUN_INSERT_NET_COPYRIGHT] } return 
    catch "set license [open [netedit getenv SCIRUN_SRCDIR]/LICENSE]"
    if { ![info exists license] } return
    while { ![eof $license] } {
	puts $out "\# [gets $license]"
    }
    close $license
}


proc init_DATADIR_and_DATASET {} {
    uplevel 1 sourceSettingsFile
    upvar 1 DATADIR datadir DATASET dataset DATAFILE datafile
    set datadir [netedit getenv SCIRUN_DATA]
    set dataset [netedit getenv SCIRUN_DATASET]
    set datafile [netedit getenv SCIRUN_DATAFILE]
    netedit setenv SCIRUN_NET_SUBSTITUTE_DATADIR true
}
    
proc backupNetwork { } {
  # check if we have a filename and if so save it to #filename#
  global netedit_savefile
  if { [file exists $netedit_savefile] } {
      set root_filename [lindex [file split $netedit_savefile] end]

      # don't save back ups of back up files as ##filename##!!!
      if {[string index $root_filename 0] == "#" && 
	  [string index $root_filename end] == "#"} {
	      set root_filename [string range $root_filename 1 end-1]
      }

      set current_path [lrange [file split $netedit_savefile] 0 end-1]
      if {[llength $current_path] > 0} {
	  set current_path [eval file join [lrange [file split $netedit_savefile] 0 end-1]]
      } else {
	  # net in current directory so we need something to the path
	  # so the file joins work
	  set current_path [pwd]
      }

      # First attempt to save it to same directory as netedit_savefile
      if {[file writable $current_path]} {
	  set dest [eval file join $current_path \#$root_filename\#]
	  writeNetwork $dest
      } else {
	  # else save to home/SCIRun directory as nedetit_savefile
	  set src "[file split [netedit getenv HOME]]"
	  set src [lappend src SCIRun \#$root_filename\#]
	  set dest [eval file join $src]
	  writeNetwork $dest
      }
  } else {
      # Attempt to write to #MyNetwork.srn# in current directory
      set src  "[file split [pwd]] MyNetwork.srn"
      set src [eval file join $src]
      set parts [file split $src]
      set parts [lreplace $parts end end \#[lindex $parts end]\#]
      set dir [pwd]

      if {[file writable $dir]} {
          set dest [eval file join $parts]
	  writeNetwork $dest
      } else {
	  # Else write to home/SCIRun directory
	  set src "[file split [netedit getenv HOME]]"
	  set src [lappend src SCIRun \#MyNetwork.srn\#]
	  set dest [eval file join $src]
	  writeNetwork $dest
      }
  }
}

proc writeNetwork { filename { subnet 0 } } {
    # if the file already exists, back it up to "filename.bak"
    if { [file exists $filename] &&
	 [string index $filename 0] != "#" &&
	 [string index $filename end] != "#"} {
	set src  "[file split [pwd]] [file split ${filename}]"
	set src [eval file join $src]
	set parts [file split $src]
	set parts "[lreplace $parts end end [lindex $parts end]].bak"
	set dest [eval file join $parts]
	catch [file rename -force $src $dest]
    }

    netedit start-net-doc $filename v[netedit getenv SCIRUN_VERSION]
    set out stdout
    
    #maybeWriteTCLStyleCopyright $out
    maybeWrite_init_DATADIR_and_DATASET $out
    genSubnetScript $subnet ""

    netedit write-net-doc
}


# Numerically compares two version strings and returns:
#  -1 if ver1 < ver2,
#   0 if ver1 == ver2
#   1 if ver1 > ver2
proc compareVersions { ver1 ver2 } {
    set v1 [split $ver1 .]
    set v2 [split $ver2 .]
    set l1 [llength $v1]
    set l2 [llength $v2]
    set len [expr ($l1 > $l2) ? $l1 : $l2]
    for { set i 0 } {$i < $len} {incr i} {
	set n1 -1
	set n2 -1
	if {$i < $l1} {
	    set n1 [lindex $v1 $i]
	}
	if {$i < $l2} {
	    set n2 [lindex $v2 $i]
	}
	if { $n1 < $n2 } {
	    return -1
	}
	if { $n2 < $n1 } {
	    return 1
	}
    }
    return 0
}

proc txt { args } {
    return [join $args \n]
}

proc in_power_app {} {
    return [winfo exists .standalone]
}
