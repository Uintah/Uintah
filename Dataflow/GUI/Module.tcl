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


global CurrentlySelectedModules
set CurrentlySelectedModules ""

global startX
set startX 0

global startY
set startY 0

global undoList
set undoList ""

global redoList
set redoList ""

itcl_class Module {
   
    method modname {} {
	return [string range $this [expr [string last :: $this] + 2] end]
    }
			
    constructor {config} {
	global $this-gui-x $this-gui-y
        set $this-gui-x -1
        set $this-gui-y -1

	
	# these live in parallel temporarily
	global $this-notes Notes
	if ![info exists $this-notes] { set $this-notes "" }
	trace variable $this-notes w "syncNotes [modname]"
	
	
	global $this-backlog
	set $this-backlog {}
    }
    
    destructor {
	set w .mLogWnd[modname]
	if [winfo exists $w] {
	    destroy $w
	}

	eval unset [info vars $this-*]
	destroy $this
    }
    
    method set_defaults {} { 
    }

    method append_log_msg {msg tag} {
        set $this-backlog [lappend $this-backlog [list "$msg" $tag]]
        append_log_aux $msg $tag
    }

    method append_log_aux {msg tag} {
 	set ww .mLogWnd[modname]
	if {[winfo exists $ww.log.txt]} {
	    $ww.log.txt config -state normal
	    $ww.log.txt insert end "$msg" $tag
	    $ww.log.txt config -state disabled
	}
    }

    public name
    protected min_text_width 0
    protected make_progress_graph 1
    protected make_time 1
    protected graph_width 50
    protected old_width 0
    protected indicator_width 15
    protected initial_width 0
    protected done_building_icon 0
    protected progress_bar_is_mapped 0
    protected time_is_mapped 0
    # flag set when the module is compiling
    protected compiling_p 0
    # flag set when the module has all incoming ports blocked
    public state "NeedData" {$this update_state}
    public msg_state "Reset" {$this update_msg_state}
    public progress 0 {$this update_progress}
    public time "00.00" {$this update_time}
    public isSubnetModule 0
    public subnetNumber 0

    method get_msg_state {} {
	return $msg_state
    }

    method compiling_p {} { return $compiling_p }
    method set_compiling_p { val } { 
	set compiling_p $val	
	setColorAndTitle
        if {[winfo exists .standalone]} {
	    app indicate_dynamic_compile [modname] [expr $val?"start":"stop"]
	}
    }

    method name {} {
	return $name
    }
    
    method set_state {st t} {
	set state $st
	set time $t
	update_state
	update_time
	
	# on windows, an update idletasks here causes hangs.  Ignoring
	# it, however, causes subtle display inconsistencies, so do 
	# everywhere but on windows
        set ostype [netedit getenv OS]
        if { ![string equal $ostype "Windows_NT"] } {
	    update idletasks
	}
    }

    method set_msg_state {st} {
	set msg_state $st
	update_msg_state
	# on windows, an update idletasks here causes hangs.  Ignoring
	# it, however, causes subtle display inconsistencies, so do 
	# everywhere but on windows
        set ostype [netedit getenv OS]
        if { ![string equal $ostype "Windows_NT"] } {
	    update idletasks
	}
    }

    method set_progress {p t} {
	set progress $p
	set time $t
	update_progress
	update_time
	update idletasks
    }

    method set_title {name} {
	# index points to the second "_" in the name, after package & category
	set index [string first "_" $name [expr [string first "_" $name 0]+1]]
	return [string range $name [expr $index+1] end]
    }

    method initialize_ui { {my_display "local"} } {
        $this ui
	if {[winfo exists .ui[modname]]!= 0} {
	    set w .ui[modname]

	    SciRaise $w

	    wm title $w [set_title [modname]]
	}
    }

    # Brings the UI window to near the mouse.
    method fetch_ui {} {
	set w .ui[modname]
	# Mac window manager top window bar height is 22 pixels (at
	# least on my machine.)  Because the 'winfo y' command does
	# not take this into account (at least on the Mac, need to
	# check PC), we need to do so explicitly. -Dav
	set wm_border_height 22
	if {[winfo exists $w] != 0} {
	    global $this-gui-x $this-gui-y
	    set $this-gui-x [winfo x $w]
	    set $this-gui-y [expr [winfo y $w] - $wm_border_height]
	    moveToCursor $w
	    # Raise the window
	    initialize_ui
	}
    }

    # Returns the UI window to where the user had originally placed it.
    method return_ui {} {
	set w .ui[modname]
	if {[winfo exists $w] != 0} {
            global $this-gui-x $this-gui-y
	    wm geometry $w +[set $this-gui-x]+[set $this-gui-y]
	    # Raise the window
	    initialize_ui
	}
    }

    method have_ui {} {
	return [llength [$this info method ui]]
    }

    #  Make the modules icon on a particular canvas
    method make_icon {modx mody { ignore_placement 0 } } {
	global Disabled Subnet Color ToolTipText
	set done_building_icon 0
	set Disabled([modname]) 0
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set minicanvas $Subnet(Subnet$Subnet([modname])_minicanvas)
	
	set modframe $canvas.module[modname]
	frame $modframe -relief raised -borderwidth 3 
	
	bind $modframe <1> "moduleStartDrag [modname] %X %Y 0"
	bind $modframe <B1-Motion> "moduleDrag [modname] %X %Y"
	bind $modframe <ButtonRelease-1> "moduleEndDrag [modname] %X %Y"
	bind $modframe <Control-Button-1> "moduleStartDrag [modname] %X %Y 1"
	bind $modframe <3> "moduleMenu %X %Y [modname]"
	
	frame $modframe.ff
	set p $modframe.ff
	pack $p -side top -expand yes -fill both -padx 5 -pady 6

	global ui_font modname_font time_font
	if {[have_ui]} {
	    button $p.ui -text "UI" -borderwidth 2 -font $ui_font \
		-anchor center -command "$this initialize_ui"
	    pack $p.ui -side left -ipadx 5 -ipady 2
	    Tooltip $p.ui $ToolTipText(ModuleUI)
	}

	# Make the title
	label $p.title -text "$name" -font $modname_font -anchor w
	pack $p.title -side [expr $make_progress_graph?"top":"left"] \
	    -padx 2 -anchor w
	bind $p.title <Map> "$this setDone"

	# Make the time label
	if {$make_time} {
	    label $p.time -text "00.00" -font $time_font
	    pack $p.time -side left -padx 2
	    Tooltip $p.time $ToolTipText(ModuleTime)
	    setGlobal $this-time_mapped 0
	    bind $p.time <Map> "setGlobal $this-time_mapped 1; $this setDone"
	} else {
	    setGlobal $this-time_mapped 1
	}

	# Make the progress graph
	if {$make_progress_graph} {
	    frame $p.inset -relief sunken -height 4 \
		-borderwidth 2 -width $graph_width
	    pack $p.inset -side left -fill y -padx 2 -pady 2
	    frame $p.inset.graph -relief raised \
		-width 0 -borderwidth 2 -background green
	    Tooltip $p.inset $ToolTipText(ModuleProgress)
	    setGlobal $this-progress_mapped 0
	    bind $p.inset <Map> "setGlobal $this-progress_mapped 1; $this setDone"
	} else {
	    setGlobal $this-progress_mapped 1
	}

	# Make the message indicator
	frame $p.msg -relief sunken -height 15 -borderwidth 1 \
	    -width [expr $indicator_width+2]
	pack $p.msg -side [expr $make_progress_graph?"left":"right"] \
	    -padx 2 -pady 2
	frame $p.msg.indicator -relief raised -width 0 -height 0 \
	    -borderwidth 2 -background blue
	bind $p.msg.indicator <Button-1> "$this displayLog"
	Tooltip $p.msg.indicator $ToolTipText(ModuleMessageBtn)

	update_msg_state
	update_progress
	update_time

	# compute the placement of the module icon
	if { !$ignore_placement } {
	    set pos [findModulePosition $Subnet([modname]) $modx $mody]
	} else {
	    set pos [list $modx $mody]
	}
	
	set pos [eval clampModuleToCanvas $pos]
	
	# Stick it in the canvas
	$canvas create window [lindex $pos 0] [lindex $pos 1] -anchor nw \
	    -window $modframe -tags "module [modname]"
	set pos [scalePath $pos]
	$minicanvas create rectangle [lindex $pos 0] [lindex $pos 1] \
	    [expr [lindex $pos 0]+6] [expr [lindex $pos 1]+2] \
	    -outline {} -fill $Color(Basecolor) -tags "[modname] module"

	# Create, draw, and bind all input and output ports
	drawPorts [modname]
	
	# create the Module Menu
	menu $p.menu -tearoff false -disabledforeground white

	Tooltip $p $ToolTipText(Module)

	bindtags $p [linsert [bindtags $p] 1 $modframe]
	bindtags $p.title [linsert [bindtags $p.title] 1 $modframe]
	bindtags $p.msg [linsert [bindtags $p.msg] 1 $modframe]
	bindtags $p.msg.indicator [linsert [bindtags $p.msg.indicator] 1 $modframe]
	if {$make_time} {
	    bindtags $p.time [linsert [bindtags $p.time] 1 $modframe]
	}
	if {$make_progress_graph} {
	    bindtags $p.inset [linsert [bindtags $p.inset] 1 $modframe]
	}

	# If we are NOT currently running a script... ie, not loading the net
	# from a file
	if ![string length [info script]] {
	    unselectAll
	    global CurrentlySelectedModules
	    set CurrentlySelectedModules "[modname]"
	}
	
	fadeinIcon [modname]
    }
    # end make_icon
    
    method setColorAndTitle { { color "" } args} {
	global Subnet Color Disabled
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set m $canvas.module[modname]
	if { ![winfo exists $m] } return
	if ![string length $color] {
	    set color $Color(Basecolor)
	    if { [$this is_selected] } { set color $Color(Selected) }
	    setIfExists disabled Disabled([modname]) 0
	    if { $disabled } { set color [blend $color $Color(Disabled)] }
	}
	if { $compiling_p } { set color $Color(Compiling) }
	if { ![llength $args] && [isaSubnetIcon [modname]] } {
	    set args $Subnet(Subnet$Subnet([modname]_num)_Name)
	}
	$m configure -background $color
	$m.ff configure -background $color
	if {[$this have_ui]} { $m.ff.ui configure -background $color }
	if {$make_time} { $m.ff.time configure -background $color }
	if {$isSubnetModule} { $m.ff.subnet configure -background $color }
	if {![llength $args]} { set args $name }
	$m.ff.title configure -text "$args" -justify left -background $color
	if {$isSubnetModule} { 
	    $m.ff.type configure -text $Subnet(Subnet$Subnet([modname]_num)_Instance) \
		-justify left -background $color
	}
	    
    }

    method addSelected {} {
	if {![$this is_selected]} { 
	    global CurrentlySelectedModules
	    lappend CurrentlySelectedModules [modname]
	    setColorAndTitle
	}
    }    

    method removeSelected {} {
	if {[$this is_selected]} {
	    #Remove me from the Currently Selected Module List
	    global CurrentlySelectedModules
	    listFindAndRemove CurrentlySelectedModules [modname]
	    setColorAndTitle
	}
    }
    
    method toggleSelected {} {
	if [is_selected] removeSelected else addSelected
    }
    
    method update_progress {} {
	if { !($progress >= 0) } { set progress 0 }
	set width [expr int($progress*($graph_width-4))]
	if {!$make_progress_graph || $width == $old_width } return
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set graph $canvas.module[modname].ff.inset.graph
	if {$width == 0} { 
#	    place forget $graph
	} elseif { [winfo exists $graph] } {
	    $graph configure -width $width
	    if {$old_width == 0} { place $graph -relheight 1 -anchor nw }
	}
	set old_width $width
    }
	
    method update_time {} {
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set timeframe $canvas.module[modname].ff.time
	if { !$make_time || ![winfo exists $timeframe] } return

	if {$state == "JustStarted"} {
	    set tstr " ?.??"
	} else {
	    if {$time < 60} {
		set secs [expr int($time)]
		set frac [expr int(100*($time-$secs))]
		set tstr [format "%2d.%02d" $secs $frac]
	    } elseif {$time < 3600} {
		set mins [expr int($time/60)]
		set secs [expr int($time-$mins*60)]
		set tstr [format "%2d:%02d" $mins $secs]
	    } else {
		set hrs [expr int($time/3600)]
		set mins [expr int($time-$hrs*3600)]
		set tstr [format "%d::%02d" $hrs $mins]
	    }
	}
	$timeframe configure -text $tstr
    }

    method update_state {} {
	if {!$make_progress_graph} return
	if {$state == "JustStarted 1123"} {
	    set progress 0.5
	    set color red
	} elseif {$state == "Executing"} {
	    set progress 0
	    set color red
	} elseif {$state == "NeedData"} {
	    set progress 1
	    set color yellow
	} elseif {$state == "Completed"} {
	    set progress 1
	    set color green
	} else {
	    set width 0
	    set color grey75
	    set progress 0
	}

	if { [winfo exists .standalone] } {
	    app update_progress [modname] $state
	}
	
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set graph $canvas.module[modname].ff.inset.graph
	if { [winfo exists $graph] } {
	    $canvas.module[modname].ff.inset.graph configure -background $color
	}
	update_progress
    }

    method update_msg_state {} { 
	if {$msg_state == "Error"} {
	    set p 1
	    set color red
	} elseif {$msg_state == "Warning"} {
	    set p 1
	    set color yellow
	} elseif {$msg_state == "Remark"} {
	    set p 1
	    set color blue
	}  elseif {$msg_state == "Reset"} {
	    set p 1
	    set color grey75
	} else {
	    set p 0
	    set color grey75
	}
	global Subnet
	set number $Subnet([modname])
	set canvas $Subnet(Subnet${number}_canvas)
	set indicator $canvas.module[modname].ff.msg.indicator
	if { [winfo exists $indicator] } {
	    $indicator configure -width $indicator_width -background $color
	}

	if $number {
	    SubnetIcon$number update_msg_state
	}

	if {[winfo exists .standalone]} {
	    app indicate_error [modname] $msg_state
	}
	
    }

    method get_x {} {
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	return [lindex [$canvas coords [modname]] 0]
    }

    method get_y {} {
	global Subnet
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	return [lindex [$canvas coords [modname]] 1]
    }

    method is_selected {} {
	global CurrentlySelectedModules
	return [expr ([lsearch $CurrentlySelectedModules [modname]]!=-1)?1:0]
    }

    method displayLog {} {
	if [$this is_subnet] return
	set w .mLogWnd[modname]
	
	if {[winfo exists $w]} {
	    SciRaise $w
	    return
	}
	
	# Create the window (immediately withdraw it to avoid flicker).
	toplevel $w; wm withdraw $w
	update

	append t "Log for " [modname]
	set t "$t -- pid=[$this-c getpid]"
	wm title $w $t
	
	frame $w.log
	# Create the txt in the "disabled" state so that a user cannot type into the text field.
	# Also, must give it a width and a height so that it will be allowed to automatically
	# change the width and height (go figure?).
	text $w.log.txt -relief sunken -bd 2 -yscrollcommand "$w.log.sb set" -state disabled \
	    -height 2 -width 10
	scrollbar $w.log.sb -relief sunken -command "$w.log.txt yview"
	pack $w.log.txt -side left -padx 5 -pady 2 -expand 1 -fill both

        # Set up our color tags.
	$w.log.txt config -state normal
	$w.log.txt tag configure red -foreground red
	$w.log.txt tag configure blue -foreground blue
	$w.log.txt tag configure yellow -foreground yellow
	$w.log.txt tag configure black -foreground "grey20"
	$w.log.txt config -state disabled

        foreach thingy [set $this-backlog] {
            append_log_aux [lindex $thingy 0] [lindex $thingy 1]
        }

	pack $w.log.sb -side left -padx 0 -pady 2 -fill y

	frame $w.fbuttons 
	# TODO: unregister only for streams with the supplied output
	button $w.fbuttons.clear -text "Clear" -command "$this clearStreamOutput"
	button $w.fbuttons.ok    -text "Close" -command "wm withdraw $w"
        bind $w <Escape> "wm withdraw $w"
	
	Tooltip $w.fbuttons.ok "Close this window.  The log is not effected."
	TooltipMultiline $w.fbuttons.clear \
            "If the log indicates anything but an error, the Clear button\n" \
            "will clear the message text from the log and reset the warning\n" \
            "indicator to No Message.  Error messages cannot be cleared -- you\n" \
	    "must fix the problem and re-execute the module."

	pack $w.fbuttons.clear $w.fbuttons.ok -side left -expand yes -fill both \
	    -padx 5 -pady 5 -ipadx 3 -ipady 3
	pack $w.log      -side top -padx 5 -pady 2 -fill both -expand 1
	pack $w.fbuttons -side top -padx 5 -pady 2 -fill x

	wm minsize $w 450 150

	# Move window to cursor after it has been created.
	moveToCursor $w "leave_up"

    }

    method clearStreamOutput { } {
	# Clear the text widget 
	set w .mLogWnd[modname]

	if {! [winfo exists $w]} {
	    return
	}

        $w.log.txt config -state normal
	$w.log.txt delete 0.0 end
        $w.log.txt config -state disabled

        set $this-backlog {}
	# Clear the module indicator color if
        # not in an error state
	if {$msg_state != "Error"} {
	    set_msg_state "Reset"
	}
    }
    
    method destroyStreamOutput {w} {
	destroy $w
    }


    method resize_icon {} {
	if { !$done_building_icon } return

	global Subnet port_spacing modname_font
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)

	set text_widget $canvas.module[modname].ff.title
	set text_width [font measure $modname_font $name]
	set text_diff [expr $text_width - $min_text_width]

	set iports [portCount "[modname] 0 i"]
	set oports [portCount "[modname] 0 o"]
	set nports [expr $oports>$iports?$oports:$iports]
	set ports_width [expr 8+$nports*$port_spacing] 
	set port_diff [expr $ports_width - $initial_width]
	set diff [expr $port_diff > $text_diff ? $port_diff : $text_diff]

	if { $diff > 0 } {
	    $canvas itemconfigure [modname] -width [expr $initial_width+$diff]
	} else {
	    $canvas itemconfigure [modname] -width $initial_width
	}
	    
    }


    method setDone {} {
	#module already mapped to the canvas
	upvar \#0 $this-progress_mapped progress_mapped
	upvar \#0 $this-time_mapped time_mapped
	if { $done_building_icon } return
	if { [info exists progress_mapped] && !$progress_mapped } return
	if { [info exists time_mapped] && !$time_mapped } return
	set done_building_icon 1
	    
	global Subnet IconWidth modname_font
	set canvas $Subnet(Subnet$Subnet([modname])_canvas)
	set initial_width [winfo width $canvas.module[modname]]
	set progress_width 0
	set time_width 0
	if {$make_progress_graph} {
	    set progress_width [winfo width $canvas.module[modname].ff.inset]
	}
	if {$make_time} {
	    set time_width [winfo width $canvas.module[modname].ff.time]
	}
	set min_text_width [expr $progress_width+$time_width]
	set text_width [winfo width $canvas.module[modname].ff.title]
	if { $min_text_width < $text_width } {
	    set min_text_width $text_width
	}
	
	resize_icon
	drawNotes [modname]
	drawConnections [portConnections "[modname] all o"]
    }

    method execute {} {
	$this-c needexecute
    }
    
    method is_subnet {} {
	return $isSubnetModule
    }

    method compile_error { filename } {
	set ccfile [netedit getenv SCIRUN_ON_THE_FLY_LIBS_DIR]/${filename}cc
	if { ![file exists $ccfile] || ![file readable $ccfile] } return
	
       	set w .text[modname]
	# create the window
	if { [winfo exists $w] } {
	    destroy $w
	}
	toplevel $w
	wm title $w "$name failed to compile"
	
	# create frame to display source code
	frame $w.f1
	text $w.f1.txt -relief sunken -wrap char \
	    -bd 2 -yscrollcommand "$w.f1.sb set" \
	    -font -*-fixed-*-*-*-*-*-120-*-*-*-*-*-1

	scrollbar $w.f1.sb -relief sunken -command "$w.f1.txt yview"
	pack $w.f1.sb -side right -padx 5 -pady 5 -fill y
	pack $w.f1.txt -side left -padx 5 -pady 5 -expand 1 -fill both 

	# create button to close dialog window
	frame $w.fbuttons 
	button $w.fbuttons.ok -text "Close" -command "destroy $w"
	
	pack $w.fbuttons.ok -side top -padx 5 -pady 5
	pack $w.fbuttons -side bottom     
	pack $w.f1 -side top -padx 5 -pady 2 -expand 1 -fill both
			
	# open the cc file and append its contents into the text widget
	set cc [open $ccfile]
	set i 0
	while { ![eof $cc] } {
	    set line [gets $cc]
	    incr i
	    set ln [format "%4d" $i]
	    $w.f1.txt insert end "${ln}: $line\n"
	    # tag line number region for later higlighting
	    $w.f1.txt tag add linenum $i.0 $i.5
	    
	}
	close $cc
	# highlight the line numbers text in blue
	$w.f1.txt tag configure linenum -foreground blue

	# if no log file exists, we are done, so return
	set logfile [netedit getenv SCIRUN_ON_THE_FLY_LIBS_DIR]/${filename}log
	if { ![file exists $logfile] || ![file readable $logfile] } return

	# Create extra widgets to display error messages for each line
	frame $w.f2
	label $w.f2.label -anchor w \
	    -text "Mouse over highligted line for compiler message:"
	text $w.f2.txt -relief sunken -wrap char -height 6 \
	    -bd 2 -yscrollcommand "$w.f2.sb set" \
	    -font -*-fixed-*-*-*-*-*-120-*-*-*-*-*-1		
	scrollbar $w.f2.sb -relief sunken -command "$w.f2.txt yview"
	pack $w.f2.label -side top -padx 0 -pady 0 -expand 1 -fill x
	pack $w.f2.sb -side right -padx 5 -fill y
	pack $w.f2.txt -side left -padx 5 -pady 0 -expand 1 -fill both 
	pack $w.f2 -side top -padx 5 -pady 2 -expand 1 -fill both

	# iterate over the log file
	set log [open $logfile]
	while { ![eof $log] } {
	    set line [gets $log]
	    set line [split $line :]
	    set num [lindex $line 1]
	    # if the log file has a line number in it that refers
	    # to this file, then highlight that line
	    if { ([string first $filename [lindex $line 0]] == 0) && \
		     [string is integer $num] } {
		# Tag as warning if message starts with ' warning'
		if { [string first " warning" [lindex $line 2]] == 0} {
		    $w.f1.txt tag add warnlines $num.6 $num.end
		} else { ; # otherwise, assume its an error
		    $w.f1.txt tag add errorlines $num.6 $num.end
		}
		# Tag the source display's line as being an error
		# used to bind mouse events when user mouses-over line
		$w.f1.txt tag add error$num $num.6 $num.end
		
		# if this isn't the first error for this line, append new-line
		if { [info exists errortext($num)] } {
		    append errortext($num) "\n"
		}
		# append the first line of the error message to our cache
		append errortext($num) "[join [lrange $line 2 end] ":"]\n"
		set done 0
		while {!$done} {
		    # this line may not be an error message, but we
		    # dont know until its read, save our position first
		    set lastpos [tell $log]
		    # get the line
		    set line [gets $log]
		    # if the line is indented, assume its part of the error msg
		    if { [string first "   " $line] == 0 } {
			append errortext($num) "$line\n"
		    } else { ;# else assume its not, and undo the last read
			set done 1
			seek $log $lastpos ; # jumps to position before read
		    }
		}

		# bind mouse-over event to display error text in bottom textbox
		$w.f1.txt tag bind error$num <Enter> \
			 "$w.f2.label configure -text \"Compiler message for line $num:\"
                          $w.f2.txt delete 0.0 end       
                          $w.f2.txt insert end \"$errortext($num)\""


	    }
	}
	close $log
	# highlight the source code error lines in light red background
	$w.f1.txt tag configure errorlines -background "\#FF4444"
	# highlight the source code warning lines in light yellow background
	$w.f1.txt tag configure warnlines -background yellow3
	# Make the mouse cursor change when over highlighed source code
	foreach tag {errorlines warnlines} {
	    $w.f1.txt tag bind $tag <Enter> \
		"$w.f1.txt configure -cursor question_arrow"
	    $w.f1.txt tag bind $tag <Leave> \
		"$w.f1.txt configure -cursor xterm"
	}
    }

    # writeStateToScript
    # Called from genSubnetScript, it will append the TCL
    # commands needed to initialize this module's variables
    # after it is created.  This is located here in the Module class
    # so sub-classes (like SCIRun_Render_Viewer) can specialize
    # the variables they write out
    #
    # 'scriptVar' is the name of the TCL variable one level
    # up that we will append our commands to 
    # 'prefix' is a prefix written for all for the variables
    # 'tab' is the indent string to make it look pretty
    method writeStateToScript { scriptVar prefix { tab "" }} {
	if [isaSubnetIcon [modname]] return
	upvar 1 $scriptVar script
	set module [modname]
	set write_vars ""
	set modstr [join [modulePath $module] ->]

	global ModuleSavedVars ModuleSubstitutedVars
	if { [info exists ModuleSavedVars($module)] } {
	    set classname [join [modulePath $module] _]
	    foreach var $ModuleSavedVars($module) {
		if { ![isaDefaultValue $module $var $classname] } {
		    lappend write_vars $var
		}
	    }
	}
	
	if { [llength $write_vars] } {
	    # Write the comment line for this modules GUI values
	    append script "\n"
	    append script "${tab}\# Set GUI variables for the $modstr Module\n"
	    foreach var $write_vars {
		upvar \#0 $module-$var val
		set varname "${prefix}-${var}"
		if { [llength $varname] > 1 } {
		    set varname \"${varname}\"
		}

		if { [info exists ModuleSubstitutedVars($module)] && \
			 [lsearch $ModuleSubstitutedVars($module) $var]!=-1} {
		    append script "${tab}set $varname "
		    append script "\"[subDATADIRandDATASET $val]\"\n"
		} else {
		    if { ![string is integer $val] } {
			set failed [catch "set num [format %.[string length $val]e $val]"]
			if { !$failed } {
			    set failed [catch "set num [expr $num]"]
			}
			if { !$failed } {
			    append script "${tab}set $varname \{$num\}\n"
			    continue
			}
		    }
		    append script "${tab}set $varname \{${val}\}\n"
		}
	    }
	}
	
	# Write command to open GUI on load if it was open on save
	if [windowIsMapped .ui$module] {
	    append script "\n${tab}\# Open the $modstr UI\n"
	    append script "${tab}${prefix} initialize_ui\n"
	}	
    }
}   

proc fadeinIcon { modid { seconds 0.333 } { center 0 }} {
    if [llength [info script]] {
	$modid setColorAndTitle
	return
    }
    global Color Subnet FadeInID
    if $center {
	set canvas $Subnet(Subnet$Subnet($modid)_canvas)
	set bbox [$canvas bbox $modid]
	if { [lindex $bbox 0] < [$canvas canvasx 0] ||
	     [lindex $bbox 1] < [$canvas canvasy 0] ||
	     [lindex $bbox 2] > [$canvas canvasy [winfo width $canvas]] ||
	     [lindex $bbox 3] > [$canvas canvasy [winfo height $canvas]] } {
	    set modW [expr [lindex $bbox 2] - [lindex $bbox 0]]
	    set modH [expr [lindex $bbox 3] - [lindex $bbox 1]]
	    set canScroll [$canvas cget -scrollregion]
	    set x [expr [lindex $bbox 0] - ([winfo width  $canvas] - $modW)/2]
	    set y [expr [lindex $bbox 1] - ([winfo height $canvas] - $modH)/2]
	    set x [expr $x/([lindex $canScroll 2] - [lindex $canScroll 0])]
	    set y [expr $y/([lindex $canScroll 3] - [lindex $canScroll 1])]
	    $canvas xview moveto $x
	    $canvas yview moveto $y
	}
    }

    set frequency 24
    set period [expr double(1000.0/$frequency)]
    set t $period
    set stopAt [expr double($seconds*1000.0)]
    set dA [expr double(1.0/($seconds*$frequency))]
    set alpha $dA
	    
    if [info exists FadeInID($modid)] {
	foreach id $FadeInID($modid) {
	    after cancel $id
	}
    }
    set FadeInID($modid) ""

    $modid setColorAndTitle $Color(IconFadeStart)
    while { $t < $stopAt } {
	set color [blend $Color(Selected) $Color(IconFadeStart) $alpha]
	lappend FadeInID($modid) [after [expr int($t)] "$modid setColorAndTitle $color"]
	set alpha [expr double($alpha+$dA)]
	set t [expr double($t+$period)]
    }
    lappend FadeInID($modid) [after [expr int($t)] "$modid setColorAndTitle"]
}
	
proc moduleMenu {x y modid} {
    global Subnet mouseX mouseY
    set mouseX $x
    set mouseY $y
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)
    set menu_id "$canvas.module$modid.ff.menu"
    regenModuleMenu $modid $menu_id
    tk_popup $menu_id $x $y    
}

set SCIRunSelection {}
proc selection_handler { args } {
    global SCIRunSelection
    return $SCIRunSelection
}

proc regenModuleMenu { modid menu_id } {
    # Wipe the menu clean...
    set num_entries [$menu_id index end]
    if { $num_entries == "none" } { 
	set num_entries 0
    }
    for {set c 0} {$c <= $num_entries} {incr c } {
	$menu_id delete 0
    }

    selection handle . "selection_handler"
    global Subnet Disabled CurrentlySelectedModules
    $menu_id add command -label "$modid" -command "setGlobal SCIRunSelection $modid; selection own ."
    $menu_id add separator

    # 'Execute Menu Option
    $menu_id add command -label "Execute" -command "$modid execute"
    
    # 'Help' Menu Option
    if { ![$modid is_subnet] } {
	$menu_id add command -label "Help" -command "moduleHelp $modid"
    }
    
    # 'Notes' Menu Option
    $menu_id add command -label "Notes" \
	-command "notesWindow $modid notesDoneModule"

    # 'Destroy' module Menu Option
    $menu_id add command -label "Destroy" \
	-command "moduleDestroySelected $modid"

    # 'Duplicate' module Menu Option
    if { ![$modid is_subnet] } {
	$menu_id add command -label "Duplicate" -command "moduleDuplicate $modid"
    }

    # 'Replace' module Menu Option
    if { ![$modid is_subnet] && [moduleReplaceMenu $modid $menu_id.replace] } {
	$menu_id add cascade -label "Replace With" -menu $menu_id.replace
    }
    
    # 'Show Log' module Menu Option
    if {![$modid is_subnet]} {
	$menu_id add command -label "Show Log" -command "$modid displayLog"
    }
    
    # 'Make Sub-Network' Menu Option
    set mods [expr [$modid is_selected]?"$CurrentlySelectedModules":"$modid"]
    $menu_id add command -label "Make Sub-Network" \
	-command "createSubnetFromModules $mods"
    
    # 'Expand Sub-Network' Menu Option
    if { [$modid is_subnet] } {
	$menu_id add command -label "Expand Sub-Network" \
	    -command "expandSubnet $modid"
    }

    # 'Enable/Disable' Menu Option
    if {[llength $Subnet(${modid}_connections)]} {
	setIfExists disabled Disabled($modid) 0
	if $disabled {
	    $menu_id add command -label "Enable" \
		-command "disableModule $modid 0"
	} else {
	    $menu_id add command -label "Disable" \
		-command "disableModule $modid 1"
	}
    }

    # 'Fetch/Return UI' Menu Option
    if { ![$modid is_subnet] && [envBool SCIRUN_GUI_UseGuiFetch] } {
        $menu_id add separator
	$menu_id add command -label "Fetch UI"  -command "$modid fetch_ui"
	$menu_id add command -label "Return UI" -command "$modid return_ui"
    }
}


proc notesDoneModule { id } {
    global $id-notes Notes
    if { [info exists Notes($id)] } {
	set $id-notes $Notes($id)
    }
}

proc notesWindow { id {done ""} } {
    global Notes Color Subnet
    set w .notes$id
    if { [winfo exists $w] } { destroy $w }
    setIfExists cache Notes($id) ""
    toplevel $w
    wm title $w $id
    text $w.input -relief sunken -bd 2 -height 20
    bind $w.input <KeyRelease> \
	"set Notes($id) \[$w.input get 1.0 \"end - 1 chars\"\]"
    frame $w.b
    button $w.b.done -text "Done" \
	-command "okNotesWindow $id \"$done\""
    button $w.b.clear -text "Clear" -command "$w.input delete 1.0 end; set Notes($id) {}"
    button $w.b.cancel -text "Cancel" -command \
	"set Notes($id) \{$cache\}; destroy $w"

    setIfExists rgb Color($id) white
    button $w.b.reset -fg black -text "Reset Color" -command \
	"set Notes($id-Color) $rgb; $w.b.color configure -bg $rgb"

    setIfExists rgb Notes($id-Color) $rgb
    button $w.b.color -fg black -bg $rgb -text "Text Color" -command \
	"colorNotes $id"
    setIfExists Notes($id-Color) Notes($id-Color) $rgb

    frame $w.d -relief groove -borderwidth 2

    setIfExists Notes($id-Position) Notes($id-Position) def

    set radiobuttons { {"Default" def} {"None" none} {"Tooltip" tooltip} {"Top" n} }
    if [info exists Subnet($id)] {
	lappend radiobuttons {"Left" w}
	lappend radiobuttons {"Right" e}
	lappend radiobuttons {"Bottom" s}
    }
	
    make_labeled_radio $w.d.pos "Display:" "" left Notes($id-Position) $radiobuttons

    pack $w.input -fill x -side top -padx 5 -pady 3
    pack $w.d -fill x -side top -padx 5 -pady 0
    pack $w.d.pos
    pack $w.b -fill y -side bottom -pady 3
    pack $w.b.done $w.b.clear $w.b.cancel $w.b.reset \
	$w.b.color -side right -padx 5 -pady 5 -ipadx 3 -ipady 3
	    
    if [info exists Notes($id)] {$w.input insert 1.0 $Notes($id)}
}

proc colorNotes { id } {
    global Notes
    set w .notes$id
    networkHasChanged
    set color [tk_chooseColor -initialcolor [$w.b.color cget -bg]]
    if { [string length $color] } { 
	set Notes($id-Color) $color
	$w.b.color configure -bg $color
    }
}

proc okNotesWindow {id {done  ""}} {
    destroy .notes$id
    if { $done != ""} { eval $done $id }
}

proc disableModule { module state } {
    global Disabled CurrentlySelectedModules Subnet
    set mods [expr [$module is_selected]?"$CurrentlySelectedModules":"$module"]
    foreach modid $mods { ;# Iterate through the modules
	foreach conn $Subnet(${modid}_connections) { ;# all module connections
	    setIfExists disabled Disabled([makeConnID $conn]) 0
	    if { $state } {
		connectionDisable $conn
	    } else {
		connectionEnable $conn
	    }
	}
    }
}

proc checkForDisabledModules { args } {
    global Disabled Subnet
    set args [lsort -unique $args]
    foreach modid $args {
	if [isaSubnetEditor $modid] continue;
	# assume module is disabled
	set Disabled($modid) 1
	foreach conn $Subnet(${modid}_connections) {
	    # if connection is enabled, then enable module
	    setIfExists disabled Disabled([makeConnID $conn]) 0
	    if { !$disabled } {
		set Disabled($modid) 0
		# module is enabled, continue onto next module
		break;
	    }
	}
	if {![llength $Subnet(${modid}_connections)]} {set Disabled($modid) 0}
	$modid setColorAndTitle
    }
}


proc canvasExists { canvas arg } {
    return [expr [llength [$canvas find withtag $arg]]?1:0]
}


proc shadow { pos } {
    return [list [expr 1+[lindex $pos 0]] [expr 1+[lindex $pos 1]]]
}

proc scalePath { path } {
    set opath ""
    global SCALEX SCALEY
    foreach pt $path {
	lappend opath [expr round($pt/(([llength $opath]%2)?$SCALEY:$SCALEX))]
    } 
    return $opath
}
    

proc getModuleNotesOptions { module } {
    global Subnet Notes
    set bbox [$Subnet(Subnet$Subnet($module)_canvas) bbox $module]
    set off 2
    set xCenter [expr ([lindex $bbox 0]+[lindex $bbox 2])/2]
    set yCenter [expr ([lindex $bbox 1]+[lindex $bbox 3])/2]
    set left    [expr [lindex $bbox 0] - $off]
    set right   [expr [lindex $bbox 2] + $off]
    setIfExists pos Notes($module-Position) def
    switch $pos {
	n { return [list $xCenter [lindex $bbox 1] -anchor s -justify center] }
	s { return [list $xCenter [lindex $bbox 3] -anchor n -justify center] }
	w { return [list $left $yCenter -anchor e -justify right] }
	# east is default
	default {  return [list $right $yCenter	-anchor w -justify left]
	}

    }
}




global ignoreModuleMove 
set ignoreModuleMove 1

proc moduleStartDrag {modid x y toggleOnly} {
    global ignoreModuleMove CurrentlySelectedModules redrawConnectionList
    set ignoreModuleMove 0
    if $toggleOnly {
	$modid toggleSelected
	set ignoreModuleMove 1
	return
    }

    global Subnet startX startY lastX lastY
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)

    #raise the module icon
    raise $canvas.module$modid

    #set module movement coordinates
    set lastX $x
    set lastY $y
    set startX $x
    set startY $y
       
    #if clicked module isnt selected, unselect all and select this
    if { ![$modid is_selected] } { 
	unselectAll
	$modid addSelected
    }

    #build a connection list of all selected modules to draw conns when moving
    set redrawConnectionList ""
    foreach csm $CurrentlySelectedModules {
	eval lappend redrawConnectionList $Subnet(${csm}_connections)
    }
    set redrawConnectionList [lsort -unique $redrawConnectionList]
    
    #create a gray bounding box around moving modules
    if {[llength $CurrentlySelectedModules] > 1} {
	set bbox [compute_bbox $canvas]
	$canvas create rectangle  $bbox -outline black -tags tempbox
    }
}

proc moduleDrag {modid x y} {
    global ignoreModuleMove CurrentlySelectedModules redrawConnectionList
    if $ignoreModuleMove return
    global Subnet grouplastX grouplastY lastX lastY
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)
    set bbox [compute_bbox $canvas]
    # When the user tries to drag a group of modules off the canvas,
    # Offset the lastX and or lastY variable, so that they can only drag
    #groups to the border of the canvas
    set min_possibleX [expr [lindex $bbox 0] + $x - $lastX]
    set min_possibleY [expr [lindex $bbox 1] + $y - $lastY]    
    if {$min_possibleX <= 0} { set lastX [expr [lindex $bbox 0] + $x] } 
    if {$min_possibleY <= 0} { set lastY [expr [lindex $bbox 1] + $y] }
    set max_possibleX [expr [lindex $bbox 2] + $x - $lastX]
    set max_possibleY [expr [lindex $bbox 3] + $y - $lastY]
    if {$max_possibleX >= 4500} { set lastX [expr [lindex $bbox 2]+$x-4500] }
    if {$max_possibleY >= 4500} { set lastY [expr [lindex $bbox 3]+$y-4500] }

    # Move each module individually
    foreach csm $CurrentlySelectedModules {
	do_moduleDrag $csm $x $y
    }	
    set lastX $grouplastX
    set lastY $grouplastY
    # redraw connections between moved modules
    drawConnections $redrawConnectionList
    # move the bounding selection rectangle
    $canvas coords tempbox [compute_bbox $canvas]
}    

proc do_moduleDrag {modid x y} {
    networkHasChanged
    global Subnet lastX lastY grouplastX grouplastY SCALEX SCALEY botFrame
    set canvas $Subnet(Subnet$Subnet($modid)_canvas)
    set minicanvas $Subnet(Subnet$Subnet($modid)_minicanvas)

    set grouplastX $x
    set grouplastY $y
    set bbox [$canvas bbox $modid]
    
    # Canvas Window width and height
    set width  [winfo width  $canvas]
    set height [winfo height $canvas]

    # Total Canvas Scroll Region width and height
    set canScroll [$canvas cget -scrollregion]
    set canWidth  [expr double([lindex $canScroll 2] - [lindex $canScroll 0])]
    set canHeight [expr double([lindex $canScroll 3] - [lindex $canScroll 1])]
        
    # Cursor movement delta from last position
    set dx [expr $x - $lastX]
    set dy [expr $y - $lastY]

    # if user attempts to drag module off left edge of canvas
    set modx [lindex $bbox 0]
    set left [$canvas canvasx 0] 
    if { [expr $modx+$dx] <= $left } {
	if { $left > 0 } {
	    $canvas xview moveto [expr ($modx+$dx)/$canWidth]
	}
	if { [expr $modx+$dx] <= 0 } {
	    $canvas move $modid [expr -$modx] 0
	    set dx 0
	}
    }
    
    #if user attempts to drag module off right edge of canvas
    set modx [lindex $bbox 2]
    set right [$canvas canvasx $width] 
    if { [expr $modx+$dx] >= $right } {
	if { $right < $canWidth } {
	    $canvas xview moveto [expr ($modx+$dx-$width)/$canWidth]
	}
	if { [expr $modx+$dx] >= $canWidth } {
	    $canvas move $modid [expr $canWidth-$modx] 0
	    set dx 0
	} 
    }
    
    #if user attempts to drag module off top edge of canvas
    set mody [lindex $bbox 1]
    set top [$canvas canvasy 0]
    if { [expr $mody+$dy] <= $top } {
	if { $top > 0 } {
	    $canvas yview moveto [expr ($mody+$dy)/$canHeight]
	}    
	if { [expr $mody+$dy] <= 0 } {
	    $canvas move $modid 0 [expr -$mody]
	    set dy 0
	}
    }
 
    #if user attempts to drag module off bottom edge of canvas
    set mody [lindex $bbox 3]
    set bottom [$canvas canvasy $height]
    if { [expr $mody+$dy] >= $bottom } {
	if { $bottom < $canHeight } {
	    $canvas yview moveto [expr ($mody+$dy-$height)/$canHeight]
	}	
	if { [expr $mody+$dy] >= $canHeight } {
	    $canvas move $modid 0 [expr $canHeight-$mody]
	    set dy 0
	}
    }

    # X and Y coordinates of canvas origin
    set Xbounds [winfo rootx $canvas]
    set Ybounds [winfo rooty $canvas]
    set currx [expr $x-$Xbounds]

    #cursor-boundary check and warp for x-axis
    if { [expr $x-$Xbounds] > $width } {
	cursor warp $canvas $width [expr $y-$Ybounds]
	set currx $width
	set scrollwidth [$botFrame.vscroll cget -width]
	set grouplastX [expr $Xbounds + $width - 5 - $scrollwidth]
    }
    if { [expr $x-$Xbounds] < 0 } {
	cursor warp $canvas 0 [expr $y-$Ybounds]
	set currx 0
	set grouplastX $Xbounds
    }
    
    #cursor-boundary check and warp for y-axis
    if { [expr $y-$Ybounds] > $height } {
	cursor warp $canvas $currx $height
	set scrollwidth [$botFrame.hscroll cget -width]
	set grouplastY [expr $Ybounds + $height - 5 - $scrollwidth]
    }
    if { [expr $y-$Ybounds] < 0 } {
	cursor warp $canvas $currx 0
	set grouplastY $Ybounds
    }
    
    # if there is no movement to perform, then return
    if {!$dx && !$dy} { return }
    
    # Perform the actual move of the module window
    $canvas move $modid $dx $dy
    eval $minicanvas coords $modid [scalePath [$canvas bbox $modid]]
    
    drawNotes $modid
}


proc drawNotes { args } {
    global Subnet Color Notes Font ToolTipText modname_font
    set Font(Notes) $modname_font

    foreach id $args {
	setIfExists position Notes($id-Position) def	
	setIfExists color Color($id) red
	setIfExists color Notes($id-Color) $color
	setIfExists text Notes($id) ""

	set isModuleNotes 0 
	if [info exists Subnet($id)] {
	    set isModuleNotes 1
	}
	
	if $isModuleNotes {
	    if { ![info exists Subnet($id)] } return
	    set subnet $Subnet($id)
	} else {
	    set idx [lindex [parseConnectionID $id] 0]
	    if { ![info exists Subnet($idx)] } return
	    set subnet $Subnet($idx)
	}
	set canvas $Subnet(Subnet${subnet}_canvas)

	if {$position == "tooltip"} {
	    if { $isModuleNotes } {
		Tooltip $canvas.module$id $text
	    } else {
		canvasTooltip $canvas $id {$text}
	    }
	} else {
	    if { $isModuleNotes } {
		Tooltip $canvas.module$id $ToolTipText(Module)
	    } else {
		canvasTooltip $canvas $id $ToolTipText(Connection)
	    }
	}
	
	if { $position == "none" || $position == "tooltip"} {
	    $canvas delete $id-notes $id-notes-shadow
	    continue
	}

        set shadowCol [expr [brightness $color]>0.2?"black":"white"]
	
	if { ![canvasExists $canvas $id-notes] } {
	    $canvas create text 0 0 -text "" \
		-tags "$id-notes notes" -fill $color
	    $canvas create text 0 0 -text "" -fill $shadowCol \
		-tags "$id-notes-shadow shadow"
	}

	if { $isModuleNotes } { 
	    set opt [getModuleNotesOptions $id]
	} else {
	    set opt [getConnectionNotesOptions $id]
	}
	
	$canvas coords $id-notes [lrange $opt 0 1]
	$canvas coords $id-notes-shadow [shadow [lrange $opt 0 1]]    
	eval $canvas itemconfigure $id-notes [lrange $opt 2 end]
	eval $canvas itemconfigure $id-notes-shadow [lrange $opt 2 end]
	$canvas itemconfigure $id-notes	-fill $color \
	    -font $Font(Notes) -text "$text"
	$canvas itemconfigure $id-notes-shadow -fill $shadowCol \
	    -font $Font(Notes) -text "$text"
	
	if {!$isModuleNotes} {
	    $canvas bind $id-notes <ButtonPress-1> "notesWindow $id"
	    $canvas bind $id-notes <ButtonPress-2> \
		"set Notes($id-Position) none"
	} else {
	    $canvas bind $id-notes <ButtonPress-1> \
		"notesWindow $id notesDoneModule"
	    $canvas bind $id-notes <ButtonPress-2> \
		"set Notes($id-Position) none"
	}
	canvasTooltip $canvas $id-notes $ToolTipText(Notes)		
	$canvas raise shadow
	$canvas raise notes
    }
    return 1
}
    


proc moduleEndDrag {modid x y} {
    global Subnet ignoreModuleMove CurrentlySelectedModules startX startY
    if $ignoreModuleMove return
    $Subnet(Subnet$Subnet($modid)_canvas) delete tempbox
    # If only one module was selected and moved, then unselect when done
    if {([expr abs($startX-$x)] > 2 || [expr abs($startY-$y)] > 2) && \
	    [llength $CurrentlySelectedModules] == 1} unselectAll    
}

proc htmlHelp {modid} {
    global BROWSER_DONT_ASK tcl_platform
    set BROWSER_DONT_ASK 0
    set path [modulePath $modid]
    set htmlpath Dataflow/XML/[lindex $path 2].html
    if { ![string equal [lindex $path 0] SCIRun] } { 
	set htmlpath Packages/[lindex $path 0]/$htmlpath
    }
    
    set usehtml 0
    set localhtml [netedit getenv SCIRUN_SRCDIR]/$htmlpath
    if { [file exists $localhtml] && [file readable $localhtml] } {
	set url file://$localhtml
	set usehtml 1
    } elseif { ![netedit sci_system echo a | telnet -e a software.sci.utah.edu 80 2> /dev/null > /dev/null] } {
	set url http://software.sci.utah.edu/src/$htmlpath
	set usehtml 1
    }
    
    if { !$usehtml } { return 0 }

    if { $tcl_platform(os) == "Darwin" } {
	set ret_val [netedit sci_system open $url]
	return [expr $ret_val == 0]
    } elseif { $tcl_platform(platform) == "unix" } {
	set browser [auto_execok [netedit getenv BROWSER]]
	if { $browser == "" } {
	    set browsers {mozilla firefox netscape opera galeon konqueror}
	    foreach command $browsers {
		set command [auto_execok $command]
		if { $command != "" } {
		    lappend browser $command
		}
	    }
	}

	if { [llength $browser] == 0 } {
	    return 0
	}

	if { [llength $browser] == 1 } { 
	    set BROWSER_DONT_ASK 1
	    setBrowser $browser $url
	    return 1
	}
	
	set w .choosebrowser
	toplevel $w
	wm title $w "Choose Help Browser"
	label $w.text -text "Choose Browser for HTML Help:" \
	    -justify left -anchor w
	pack $w.text -side top -expand 1 -fill x -ipady 5
       	
	listbox $w.list -selectmode single
	
	foreach command $browser {
	    $w.list insert end $command
	}
	
	frame $w.f

	$w.list activate 0
	set BROWSER_DONT_ASK 1
	checkbutton $w.f.ask -variable BROWSER_DONT_ASK \
	    -text "Dont ask for help viewer again"
	pack $w.f.ask -side top -ipady 5 -fill x
	

	button $w.f.cancel -text "View Help as Text" -command "textHelp $modid"
	
	button $w.f.ok -text "Open in HTML" \
	    -command "setBrowser \[$w.list get active\] $url"

	pack $w.f.ok $w.f.cancel -side right -ipadx 5 -ipady 5 -padx 5 -pady 5
	pack $w.f -side bottom -expand 1 -fill x -padx 5 -pady 5
	pack $w.list -side top -expand 1 -fill both -padx 10
	return 1
    }
    return 0
}

proc textHelp { modid } {
    if { [winfo exists .choosebrowser] } {
	destroy .choosebrowser
    }
    netedit setenv BROWSER text
    moduleHelp $modid
    global BROWSER_DONT_ASK
    if { !$BROWSER_DONT_ASK } {
	netedit setenv BROWSER ""
    }
}

proc setBrowser { browser url } {
    if { ![openBrowser $browser $url] } { 
	if { [winfo exists .choosebrowser] } {
	    destroy .choosebrowser
	}
	global BROWSER_DONT_ASK
	if { $BROWSER_DONT_ASK } {
	    netedit setenv BROWSER $browser
	}
    }
}

    
# return 0 on success, exit code otherwise
proc openBrowser { command url } {
    set teststring [string tolower $command]
    if { 0 && [string match -nocase "*mozilla*" $command] || \
	     [string match -nocase "*firefox*" $command] || \
	     [string match -nocase "*netscape*" $command] } {
	set ping [netedit sci_system $command -remote 'ping()' 2> /dev/null > /dev/null]
	if {$ping} {
	    return [netedit sci_system $command -remote 'openurl($url)' 2> /dev/null > /dev/null &]
	}
    } 
    return [netedit sci_system $command $url 2> /dev/null > /dev/null &]
}


proc moduleHelp { modid } {
    set w .mHelpWindow[$modid name]
	
    # does the window exist?
    if [winfo exists $w] {
	SciRaise $w
	return
    }

    if { [netedit getenv BROWSER] != "text" && [htmlHelp $modid] } return
	
    # create the window
    toplevel $w
    append t "Help for " [$modid name]
    wm title $w $t
	
    frame $w.help
#    iwidgets::scrolledhtml $w.help.txt -update 0 -alink yellow -link purple
    text $w.help.txt -relief sunken -wrap word -bd 2 -yscrollcommand "$w.help.sb set"
#    $w.help.txt render [$modid-c help]
    scrollbar $w.help.sb -relief sunken -command "$w.help.txt yview"
    pack $w.help.sb -side right -padx 5 -pady 5 -fill y
    pack $w.help.txt -side left -padx 5 -pady 5 -expand 1 -fill both 


    frame $w.fbuttons 
    button $w.fbuttons.ok -text "Close" -command "destroy $w"

    pack $w.fbuttons.ok -side top -padx 5 -pady 5
    pack $w.fbuttons -side bottom     
    pack $w.help -side top -padx 5 -pady 5 -expand 1 -fill both


    $w.help.txt insert end [$modid-c help]
    $w.help.txt configure -state disabled
}

proc moduleDestroy {modid} {
    global Subnet CurrentlySelectedModules Notes Disabled
    networkHasChanged
    if [isaSubnetIcon $modid] {
	foreach submod $Subnet(Subnet$Subnet(${modid}_num)_Modules) {
	    moduleDestroy $submod
	}
    } else {
	netedit deletemodule_warn $modid
    }

    # Deleting the module connections backwards works for dynamic modules
    set connections $Subnet(${modid}_connections)
    for {set j [expr [llength $connections]-1]} {$j >= 0} {incr j -1} {
	destroyConnection [lindex $connections $j]
    }

    # Delete Icon from canvases
    $Subnet(Subnet$Subnet($modid)_canvas) delete $modid $modid-notes $modid-notes-shadow
    destroy $Subnet(Subnet$Subnet($modid)_canvas).module$modid
    $Subnet(Subnet$Subnet($modid)_minicanvas) delete $modid
    # Remove references to module is various state arrays
    listFindAndRemove Subnet(Subnet$Subnet($modid)_Modules) $modid
    listFindAndRemove CurrentlySelectedModules $modid

    # Must have the '_' on the unset, other wise modid 1* deletes 1_* and 1#_*, etc.
    array unset Subnet ${modid}_*

    array unset Disabled $modid
    array unset Notes $modid
    array unset Notes $modid-*

    $modid delete
    if { ![isaSubnetIcon $modid] } {
	proc $modid { args } { }
	netedit deletemodule $modid
    }
    
    # Kill the modules UI if it exists
    if {[winfo exists .ui$modid]} {
	destroy .ui$modid
    }
}

proc moduleDuplicate { module } {
    global Subnet
    networkHasChanged
 
    set canvas $Subnet(Subnet$Subnet($module)_canvas) 
    set canvassize [$canvas cget -scrollregion]
    set ulx [expr [lindex [$canvas xview] 0]*[lindex $canvassize 2]]
    set uly [expr [lindex [$canvas yview] 0]*[lindex $canvassize 3]]
    set bbox [$canvas bbox $module]
    set x [expr [lindex $bbox 0]-$ulx]
    set y [expr 20 + [lindex $bbox 3] - $uly]
    set Subnet(Loading) $Subnet($module)
    set newmodule [eval addModuleAtPosition [modulePath $module] $x $y]
    set Subnet(Loading) 0

    foreach connection $Subnet(${module}_connections) {
	if { [string equal [iMod connection] $module] } {
	    createConnection [lreplace $connection 2 2 $newmodule] 1 1
	}
    }

    foreach oldvar [uplevel \#0 info vars $module-*] { 
	set pos [expr [string length $module]-1]
	set newvar [string replace $oldvar 0 $pos $newmodule]
	upvar \#0 $oldvar oldval $newvar newval
	catch "set newval \{$oldval\}"
    }
    setGlobal $newmodule-progress_mapped 0
    setGlobal $newmodule-time_mapped 0
}


# moduleCompareCommand compares two module lists {package category module} and
# returns -1, 0, or 1  if the mod1 is considered 
# less than, equal to, or greater than mod2, respectively.
# Handles the configure specific sorting of SCIRun packages
proc moduleCompareCommand { mod1 mod2 } {
    set packages [split [netedit getenv SCIRUN_LOAD_PACKAGE] ,]
    set mod1p [lsearch $packages [lindex $mod1 0]]
    set mod2p [lsearch $packages [lindex $mod2 0]]
    if { $mod1p == -1 || $mod2p == -1 } {
	puts "ERROR in moduleCompareCommand, package not found"
	return -1;
    }
    
    if { $mod1p < $mod2p } { 
	return -1 
    } elseif { $mod1p > $mod2p } { 
	return 1
    }

    return [string compare -nocase [lrange $mod1 1 2] [lrange $mod2 1 2]]
}

proc findModuleReplacements { module } {
    global Subnet ModuleIPorts ModuleOPorts

    set path [modulePath $module]
    set iports ""
    set oports ""
    foreach conn $Subnet(${module}_connections) {
	if { [string equal $module [iMod conn]] } {
	    lappend iports [iNum conn]
	}	    
	if { [string equal $module [oMod conn]] } {
	    lappend oports [oNum conn]
	}
    }
    # input port are always unique: set iports [lsort -unique $iports]
    set oports [lsort -unique $oports]
    
    set candidates [array names ModuleIPorts]
    foreach pnum $iports {
	set newcandidates ""
	set ptype [lindex $ModuleIPorts($path) $pnum]
	foreach maybe $candidates {	    
	    if { [string equal $ptype [lindex $ModuleIPorts($maybe) $pnum]] } {
		lappend newcandidates $maybe
	    }
	}
	set candidates $newcandidates
    }

    foreach pnum $oports {
	set newcandidates ""
	set ptype [lindex $ModuleOPorts($path) $pnum]
	foreach maybe $candidates {	    
	    if { [string equal $ptype [lindex $ModuleOPorts($maybe) $pnum]] } {
		lappend newcandidates $maybe
	    }
	}
	set candidates $newcandidates
    }

    # Remove the module we're replacing from the replacement possibilites list
    listFindAndRemove candidates $path

    #TODO: SUBNETS

    return [lsort -command moduleCompareCommand $candidates]
}


proc moduleReplaceMenu { module menu } {
    global ModuleMenu
    # return if there is no information to put in menu
    if { ![info exists ModuleMenu] } { return 0 }
    set moduleList [findModuleReplacements $module]
    if { ![llength $moduleList] } { return 0 }
    # destroy the old menu
    if { [winfo exists $menu] } { destroy $menu }
    # create a new menu
    menu $menu -tearoff false -disabledforeground black

    set added ""
    foreach path $moduleList {
	if { [lsearch $added [lindex $path 0]] == -1 } {
	    lappend added [lindex $path 0]
	    # Add a menu separator if this package isn't the first one
	    if { [$menu index end] != "none" } {
		$menu add separator 
	    }
	    # Add a label for the Package name
	    $menu add command -label [lindex $path 0] -state disabled
	}

	set submenu $menu.menu_[join [lrange $path 0 1] _]
	if { ![winfo exists $submenu] } {
	    menu $submenu -tearoff false
	    $menu add cascade -label "  [lindex $path 1]" -menu $submenu
	}
	set command "replaceModule $module $path"
	$submenu add command -label [lindex $path 2] -command $command
    }
    update idletasks
    return 1
}


proc replaceModule { oldmodule package category module } {
    global Subnet inserting insertOffset
    set connections $Subnet(${oldmodule}_connections)
    set bbox [$Subnet(Subnet$Subnet($oldmodule)_canvas) bbox $oldmodule]
    set x [lindex $bbox 0]
    set y [lindex $bbox 1]

    foreach connection $connections {
	destroyConnection $connection 1 1 1
    }

    moduleDestroy $oldmodule 
    set inserting 1
    set insertOffset "0 0"
    set newmodule [addModuleAtPosition $package $category $module $x $y]
    set inserting 0

    foreach connection [lsort -integer -index 3 $connections] {
	
	if { [string equal [oMod connection] $oldmodule] } {
	    set connection [lreplace $connection 0 0 $newmodule]
	}

	if { [string equal [iMod connection] $oldmodule] } {
	    set connection [lreplace $connection 2 2 $newmodule]
	}
	after 100 createConnection "\{$connection\}" 1 1
    }
}


proc moduleDestroySelected { module } {
    global CurrentlySelectedModules 

    if { [llength $CurrentlySelectedModules] <= 1 } {
	moduleDestroy $module
    } else {
	foreach mnum $CurrentlySelectedModules { 
	    moduleDestroy $mnum
	}
    }
}


global Box
set Box(InitiallySelected) ""
set Box(x0) 0
set Box(y0) 0

proc startBox {canvas X Y keepselected} {
    global Box CurrentlySelectedModules
    
    set Box(InitiallySelected) $CurrentlySelectedModules
    if {!$keepselected} {
	unselectAll
	set Box(InitiallySelected) ""
    }
    #Canvas Relative current X and Y positions
    set Box(x0) [expr $X - [winfo rootx $canvas] + [$canvas canvasx 0]]
    set Box(y0) [expr $Y - [winfo rooty $canvas] + [$canvas canvasy 0]]
    # Create the bounding box graphic
    $canvas create rectangle $Box(x0) $Box(y0) $Box(x0) $Box(y0)\
	-tags "tempbox temp"    
}

proc makeBox {canvas X Y} {    
   global Box CurrentlySelectedModules
    #Canvas Relative current X and Y positions
    set x1 [expr $X - [winfo rootx $canvas] + [$canvas canvasx 0]]
    set y1 [expr $Y - [winfo rooty $canvas] + [$canvas canvasy 0]]
    #redraw box
    $canvas coords tempbox $Box(x0) $Box(y0) $x1 $y1
    # select all modules which overlap the current bounding box
    set overlappingModules ""
    set overlap [$canvas find overlapping $Box(x0) $Box(y0) $x1 $y1]
    foreach id $overlap {
	set tags [$canvas gettags $id] 
	set pos [lsearch -exact $tags "module"]
	if { $pos != -1 } {
	    set modname [lreplace $tags $pos $pos]
	    lappend overlappingModules $modname
	    if { ![$modname is_selected] } {
		$modname addSelected
	    }
	}
    }
    # remove those not initally selected or overlapped by box
    foreach mod $CurrentlySelectedModules {
	if {[lsearch $overlappingModules $mod] == -1 && \
		[lsearch $Box(InitiallySelected) $mod] == -1} {
	    $mod removeSelected
	}
    }
}

proc unselectAll {} {
    global CurrentlySelectedModules
    foreach i $CurrentlySelectedModules {
	$i removeSelected
    }
}

proc selectAll { { subnet 0 } } {
    unselectAll
    global Subnet
    foreach mod $Subnet(Subnet${subnet}_Modules) {
	$mod addSelected
    }
}


# Courtesy of the Tcl'ers Wiki (http://mini.net/tcl)
proc brightness { color } {
    foreach {r g b} [winfo rgb . $color] break
    set max [lindex [winfo rgb . white] 0]
    return [expr {($r*0.3 + $g*0.59 + $b*0.11)/$max}]
 } ;#RS, after [Kevin Kenny]

proc blend { c1 c2 { alpha 0.5 } } {
    foreach {r1 g1 b1} [winfo rgb . $c1] break
    foreach {r2 g2 b2} [winfo rgb . $c2] break
    set max [expr double([lindex [winfo rgb . white] 0])]
    set oma   [expr (1.0 - $alpha)/$max]
    set alpha [expr $alpha / $max]

    set r [expr int(255*($r1*$alpha+$r2*$oma))]
    set g [expr int(255*($g1*$alpha+$g2*$oma))]
    set b [expr int(255*($b1*$alpha+$b2*$oma))]
    return [format "\#%02x%02x%02x" $r $g $b]
} 



proc clipBBoxes { args } {
    if { [llength $args] == 0 } { return "0 0 0 0" }
    set box1 [lindex $args 0]
    set args [lrange $args 1 end]
    while { [llength $args] } {
	set box2 [lindex $args 0]
	set args [lrange $args 1 end]
	foreach i {0 1} {
	    if {[lindex $box1 $i]<[lindex $box2 $i] } {
		set box1 [lreplace $box1 $i $i [lindex $box2 $i]]
	    }
	}
	foreach i {2 3} {
	    if {[lindex $box1 $i]>[lindex $box2 $i] } {
		set box1 [lreplace $box1 $i $i [lindex $box2 $i]]
	    }
	}
	if { [lindex $box1 2] <= [lindex $box1 0] || \
	     [lindex $box1 3] <= [lindex $box1 1] } {
	    return "0 0 0 0"
	}	     
    }
    return $box1
}	    

proc findModulePosition { subnet x y } {
    # if loading the module from a network, dont change its saved position
    if { [string length [info script]] } { return "$x $y" }
    global Subnet
    set canvas $Subnet(Subnet${subnet}_canvas)
    set wid  180
    set hei  80
    set canW [expr [winfo width $canvas] - $wid]
    set canH [expr [winfo height $canvas] - $hei]
    set maxx [$canvas canvasx $canW]
    set maxy [$canvas canvasy $canH]
    set x1 $x
    set y1 $y
    set acceptableNum 0
    set overlapNum 1 ;# to make the wile loop a do-while loop
    while { $overlapNum > $acceptableNum && $acceptableNum < 10 } {
	set overlapNum 0
	foreach tagid [$canvas find overlapping $x1 $y1 \
			   [expr $x1+$wid] [expr $y1+$hei]] {
	    foreach tags [$canvas gettags $tagid] {
		if { [lsearch $tags module] != -1 } {
		    incr overlapNum
		}
	    }
	}
	if { $overlapNum > $acceptableNum } {
	    set y1 [expr $y1 + $hei/3]
	    if { $y1 > $maxy } {
		set y1 [expr $y1-$canH+10+10*$acceptableNum]
		set x1 [expr $x1 + $wid/3]
		if { $x1 > $maxx} {
		    set x1 [expr $x1-$canW+10+10*$acceptableNum]
		    incr acceptableNum
		    incr overlapNum ;# to make sure loop executes again
		}
	    }

	}
    }
    return "$x1 $y1"
}

proc clampModuleToCanvas { x1 y1 } {
    global mainCanvasWidth mainCanvasHeight
    set wid  180
    set hei  80
    if { $x1 < 0 } { set x1 0 }
    if { [expr $x1+$wid] > $mainCanvasWidth } { 
	set x1 [expr $mainCanvasWidth-$wid]
    }
    if { $y1 < 0 } { set y1 0 }
    if { [expr $y1+$hei] > $mainCanvasHeight } { 
	set y1 [expr $mainCanvasHeight-$hei]
    }
    return "$x1 $y1"
}

	

proc isaSubnet { modid } {
    return [expr [string first Subnet $modid] == 0]
}

proc isaSubnetIcon { modid } {
    return [expr [string first SubnetIcon $modid] == 0]
}

proc isaSubnetEditor { modid } {
    return [expr ![isaSubnetIcon $modid] && [isaSubnet $modid]]
}


trace variable Notes wu notesTrace
trace variable Disabled wu disabledTrace

proc syncNotes { Modname VarName Index mode } {
    global Notes $VarName
    set Notes($Modname) [set $VarName]
}

# This proc will set Varname to the global value of GlobalName if GlobalName exists
# if GlobalName does not exist it will set Varname to DefaultVal, otherwise
# nothing is set
proc setIfExists { Varname GlobalName { DefaultVal __none__ } } {
    upvar $Varname var $GlobalName glob
#    upvar 
    if [info exists glob] {
	set var $glob
    } elseif { ![string equal $DefaultVal __none__] } {
	set var $DefaultVal
    }
}

# This proc will set Varname to the global value of GlobalName if GlobalName exists
# if GlobalName does not exist it will set Varname to DefaultVal, otherwise
# nothing is set
proc renameGlobal { Newname OldName } {
    upvar \#0 $Newname new
    upvar \#0 $OldName old
    if [info exists old] {
	set new $old
	unset old
    }
}

# This proc will unset a global variable without compaining if it doesnt exist
proc unsetIfExists { Varname } {
    upvar $Varname var
    if [info exists var] {
	unset var
    }
}


proc notesTrace { ArrayName Index mode } {
    # the next lines are to handle notes $id-Color and $id-Position changes
    set pos [string last - $Index]
    if { $pos != -1 } { set Index [string range $Index 0 [expr $pos-1]] }
    if { ![string length $Index] } return 
    networkHasChanged
    drawNotes $Index
    return 1
}

proc disabledTrace { ArrayName Index mode } {
    if ![string length $Index] return
    networkHasChanged
    global Subnet Disabled Color disableDisabledTrace
    # If disabled index is a module id, do nothing and return
    if { [info exists Subnet($Index)] } return

    # disabled is the state we just set the connection $conn to
    set conn [parseConnectionID $Index]
    setIfExists disabled Disabled($Index) 0

    set iPorts ""
    set oPorts ""
    lappend portsTodo [iPort conn] [oPort conn]
    while { [llength $portsTodo]} {
	set port [lindex $portsTodo end]
	set portsTodo [lrange $portsTodo 0 end-1]
	if { ![isaSubnet [pMod port]] } {
	    lappend [pType port]Ports $port
	} else {
	    if { [isaSubnetIcon [pMod port]] } {
		set mod Subnet$Subnet([pMod port]_num)
	    } elseif { [isaSubnetEditor [pMod port]] } {
		set mod SubnetIcon$Subnet([pMod port])
	    }
	    foreach sconn [portConnections "$mod [pNum port] [invType port]"] {
		setIfExists pathblocked Disabled([makeConnID $sconn]) 0
		if { !$pathblocked } {
		    lappend portsTodo [[pType port]Port sconn]
		}
	    }
	}
    }
	
    if { $disabled } {
	setGlobal Notes($Index-Color) $Color(ConnDisabled)
    } else {	    
	setGlobal Notes($Index-Color) $Color($Index)
    }

    foreach iPort $iPorts {
	foreach oPort $oPorts {
	    set rconn [makeConn $iPort $oPort]
	    if { $disabled } {
		netedit deleteconnection [makeConnID $rconn] 1
	    } else {
		eval netedit addconnection $rconn
	    }
	}
    }

    drawConnections [list $conn]
    $Subnet(Subnet$Subnet([oMod conn])_canvas) raise $Index
    checkForDisabledModules [oMod conn] [iMod conn]

    return 1
}




# Returns 1 if the window is mapped.  Use this function if you don't
# know whether the window exists yet.
proc windowIsMapped { w } {
    if {[winfo exists $w]} {
	if {[winfo ismapped $w]} {
	    return 1
	}
    }
    return 0
}
