#!/usr/local/bin/wish -f

if {[catch {set sci_root $env(SCI_WORK)}]} {
    puts "The environment variable SCI_WORK must be set!";
    puts "Cannot continue";
    netedit quit
}

source $sci_root/defaults.tcl
source $sci_root/devices.tcl

set modname_font "-Adobe-Helvetica-bold-R-Normal-*-*-120-75-*"
set ui_font "-Adobe-Helvetica-medium-R-Normal-*-*-120-75-*"
set time_font "-Adobe-Helvetica-medium-R-Normal-*-*-120-75-*"

proc resource {} {
}

proc makeNetworkEditor {} {

    wm minsize . 100 100
    frame .main_menu -relief raised -borderwidth 3
    pack .main_menu -fill x
    menubutton .main_menu.file -text "File" -underline 0 \
	-menu .main_menu.file.menu
    menu .main_menu.file.menu
    .main_menu.file.menu add command -label "Save..." -underline 0 \
	-command "popupSaveMenu"

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
    global sci_root
    .main_menu.help.menu add command -label "Help..." -underline 0 \
	    -command "helpPage $sci_root/help/scirun.html"

    pack .main_menu.file .main_menu.stats -side left
    pack .main_menu.help -side right

    tk_menuBar .main_menu .main_menu.file .main_menu.stats .main_menu.help

    frame .all_lists -relief groove
    pack .all_lists -side left -padx 5 -pady 5 -ipadx 2 -ipady 2 -fill y
    frame .all_lists.l -relief groove
    pack .all_lists.l -side left -padx 2 -pady 2 -fill y
    label .all_lists.l.title -text "Complete List:"
    frame .all_lists.l.f -relief groove
    pack .all_lists.l.title -anchor w
    pack .all_lists.l.f -anchor w -fill y -expand 1
    
    scrollbar .all_lists.l.f.scroll -relief sunken \
	-command ".all_lists.l.f.list yview"
    listbox .all_lists.l.f.list -yscroll ".all_lists.l.f.scroll set" \
	-relief sunken -width 20 -height 34 -exportselection false
    pack .all_lists.l.f.scroll -side right -fill y -padx 2 -expand yes -fill y
    pack .all_lists.l.f.list -side left -expand yes -fill both
    global netedit_completelist
    set netedit_completelist .all_lists.l.f.list
    
    frame .l
    pack .l -anchor w -fill x
    
    frame .l.lists -relief groove -borderwidth 4
    pack .l.lists -padx 5 -pady 5 -ipadx 2 -ipady 2 -side left -anchor w
    
    frame .l.lists.l1
    pack .l.lists.l1 -side left -padx 2 -pady 2
    label .l.lists.l1.title -text "Category"
    frame .l.lists.l1.f
    pack .l.lists.l1.title .l.lists.l1.f -anchor w
    scrollbar .l.lists.l1.f.scroll -relief sunken \
	-command ".l.lists.l1.f.list yview"
    listbox .l.lists.l1.f.list -yscroll ".l.lists.l1.f.scroll set" \
	-relief sunken -width 20 -height 4 -exportselection false
    pack .l.lists.l1.f.scroll -side right -fill y -padx 2
    pack .l.lists.l1.f.list -side left -expand yes -fill both
    
    global netedit_categorylist
    set netedit_categorylist .l.lists.l1.f.list
    
    frame .l.lists.l2
    pack .l.lists.l2 -side left -padx 2 -pady 2
    label .l.lists.l2.title -text "Modules"
    frame .l.lists.l2.f
    pack .l.lists.l2.title .l.lists.l2.f -anchor w
    scrollbar .l.lists.l2.f.scroll -relief sunken \
	-command ".l.lists.l2.f.list yview"
    listbox .l.lists.l2.f.list -yscroll ".l.lists.l2.f.scroll set" \
	-relief sunken -width 20 -height 4 -exportselection false
    pack .l.lists.l2.f.scroll -side right -fill y -padx 2
    pack .l.lists.l2.f.list -side left -expand yes -fill both
    global netedit_modulelist
    set netedit_modulelist .l.lists.l2.f.list
    
    frame .t -borderwidth 5
    pack .t -fill x
    text .t.text -relief sunken -bd 2 -yscrollcommand ".t.s set" \
	-height 3 -width 80
    scrollbar .t.s -relief sunken -command ".t.text yview"
    pack .t.s -side right -fill y -padx 4
    pack .t.text -expand yes -fill x
    global netedit_errortext
    set netedit_errortext .t.text
    
    frame .cframe -borderwidth 5
    pack .cframe -side top -expand yes -fill both
    
    frame .cframe.f -relief sunken -borderwidth 3
    canvas .cframe.f.canvas -scrollregion {0c 0c 100c 100c} \
	-xscrollcommand ".cframe.hscroll set" -yscrollcommand ".cframe.vscroll set" \
	-bg "#224488" -width 9c -height 9c
    scrollbar .cframe.vscroll -relief sunken \
	-command ".cframe.f.canvas yview"
    scrollbar .cframe.hscroll -orient horizontal -relief sunken \
	-command ".cframe.f.canvas xview"
    pack .cframe.vscroll -side right -fill y -padx 4
    pack .cframe.hscroll -side bottom -fill x -pady 4
    pack .cframe.f -expand yes -fill both
    pack .cframe.f.canvas -expand yes -fill both

    updateCategoryList
    set all_modules [netedit completelist]
    foreach i $all_modules {
	$netedit_completelist insert end $i
    }

    bind $netedit_completelist <Double-1> \
	    "addModule \[%W get \[%W nearest %y\]\]"
    bind $netedit_modulelist <Double-1> \
	    "addModule \[%W get \[%W nearest %y\]\]"
    bind $netedit_categorylist <Button-1> \
	    "showCategoryListN \[$netedit_categorylist nearest %y\]"
    global netedit_canvas
    set netedit_canvas .cframe.f.canvas
}

proc updateCategoryList {} {
    global netedit_categorylist
    $netedit_categorylist delete 0 end
    set cats [netedit catlist]
    global module_cats
    catch {unset module_cats}
    set firstcat [lindex [lindex $cats 0] 0]
    foreach i $cats {
	# The first item in the list is the category name
	set name [lindex $i 0]
	set mods [lindex $i 1]
	set module_cats($name) $mods
	$netedit_categorylist insert end $name
    }
    showCategoryList $firstcat
    $netedit_categorylist selection set 0
}

proc showCategoryList {name} {
    global netedit_modulelist
    global module_cats
    set mods $module_cats($name)
    $netedit_modulelist delete 0 end
    foreach i $mods {
	$netedit_modulelist insert end $i
    }
    $netedit_modulelist selection set 0
}

proc showCategoryListN {which} {
    global netedit_categorylist
    $netedit_categorylist selection set 0
    set name [$netedit_categorylist get $which]
    showCategoryList $name
}

proc moveModule {name} {
    
}

proc addModule {name} {
    return [addModuleAtPosition $name 10 10]
}

proc addModuleAtPosition {name xpos ypos} {
    set modid [netedit addmodule $name]
    # Create the itcl object
    if {[catch "$name $modid" problem]} {
	if {[string first "invalid command" $problem] == -1} {
	    puts "Error instantiating module: $name"
	    puts $problem
	
	    puts ""
	}
	# Use generic module
	Module $modid -name $name
    }
    $modid make_icon .cframe.f.canvas $xpos $ypos
    update idletasks
    return $modid
}

proc addConnection {omodid owhich imodid iwhich} {
    set connid [netedit addconnection $omodid $owhich $imodid $iwhich]
    set portcolor [lindex [lindex [$omodid-c oportinfo] $owhich] 0]
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
    set w .netedit_save
    if {[winfo exists $w]} {
	raise $w
	return;
    }
    toplevel $w
    makeFilebox $w netedit_savefile {netedit savenetwork $netedit_savefile} \
	"destroy $w"
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



source $sci_root/TCL/MemStats.tcl
source $sci_root/TCL/DebugSettings.tcl
source $sci_root/TCL/ThreadStats.tcl
source $sci_root/Dataflow/Module.tcl

source $sci_root/TCL/HelpPage.tcl

source $sci_root/auto.tcl
