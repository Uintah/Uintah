set basecolor gray
. configure -background $basecolor

option add *Frame*background black

option add *Button*padX 1
option add *Button*padY 1

option add *background $basecolor
option add *activeBackground $basecolor
option add *sliderForeground $basecolor
option add *troughColor $basecolor
option add *activeForeground white

option add *Scrollbar*activeBackground $basecolor
option add *Scrollbar*foreground $basecolor
option add *Scrollbar*width .35c
option add *Scale*width .35c

option add *selectBackground "white"
option add *selector red
option add *font "-Adobe-Helvetica-bold-R-Normal--*-120-75-*"
option add *highlightThickness 0

set remoteCanvasWidth    4500.0
set remoteCanvasHeight   4500.0

global chatt
global renderers
# global transferMode
global lights
global shadingType
global fog
global clients
global clientindices
global annotateItems
global remoteModuleList
global inports

proc makeUI {} {
    global chatt
    global lights
    global shadingType
    global fog

    wm minsize . 100 100
    wm geometry . 870x800+0+0

    # MAIN MENU
    frame .main_menu -relief raised -borderwidth 3
    pack .main_menu -fill x

    # File
    menubutton .main_menu.file -text "File" -underline 0 \
	-menu .main_menu.file.menu
    menu .main_menu.file.menu -tearoff false
    menu .main_menu.file.menu.new -tearoff false
    .main_menu.file.menu add command -label "Connect" -underline 0 \
	-command showConnectDialog

    .main_menu.file.menu add command -label "Quit" -underline 0 \
	    -command "ui quit"
    pack .main_menu.file -side left

    # Renderers
    menubutton .main_menu.renderers -text "Renderers" -underline 0 \
	    -menu .main_menu.renderers.menu
    menu .main_menu.renderers.menu -tearoff false
    menu .main_menu.renderers.menu.new -tearoff false

    pack .main_menu.renderers -side left

    # Compression
    menubutton .main_menu.compression -text "Compression" -underline 0 \
	    -menu .main_menu.compression.menu
    menu .main_menu.compression.menu -tearoff false
    menu .main_menu.compression.menu.new -tearoff false
    .main_menu.compression.menu add command -label "None" -underline 0 \
	    -command "ui changeCompression none"

    pack .main_menu.compression -side left

    # Transfer Mode

    menubutton .main_menu.transferMode -text "Transfer Mode" -underline 0 \
	    -menu .main_menu.transferMode.menu
    menu .main_menu.transferMode.menu -tearoff false
    menu .main_menu.transferMode.menu.new -tearoff false
    .main_menu.transferMode.menu add command -label "PTP" -underline 0 \
	    -command "ui changeTransfer PTP"

    pack .main_menu.transferMode -side left

    # Options
    menubutton .main_menu.options -text "Options" -underline 0 \
	    -menu .main_menu.options.menu
    menu .main_menu.options.menu -tearoff false
    menu .main_menu.options.menu.new -tearoff false
    .main_menu.options.menu add command -label "Remote Module Viewer" \
	    -underline 0 \
	    -command mkRemoteModuleViewer
    pack .main_menu.options -side left


    tk_menuBar .main_menu .main_menu.file .main_menu.renderers \
	    .main_menu.compression .main_menu.transferMode

    ######### Top frame ##########
    frame .top  -relief raised
    pack .top -side top

    # Left side - Client List
    frame .top.clientframe -relief ridge -borderwidth 3
    pack .top.clientframe -side left -fill y
    label .top.clientframe.listlabel -text "Client List" -fg purple
    pack .top.clientframe.listlabel -side top
    Scrolled_Listbox .top.clientframe.clientlist -width 20 -bg white
    pack .top.clientframe.clientlist -side top -expand true -fill y
    button .top.clientframe.clientrefresh -text "Refresh" -command "ui clientlistrefresh"
    pack .top.clientframe.clientrefresh -side bottom -expand true -fill x
    
    # Middle - Viewing window
    frame .top.viewframe -relief raised
    pack .top.viewframe -side left

    # Right side - lighting, shading, annotation
    frame .top.lightshade -relief flat -borderwidth 5
    pack .top.lightshade -side left

    # Lighting
    frame .top.lightshade.lighting -relief ridge -borderwidth 3
    pack .top.lightshade.lighting -side top
    label .top.lightshade.lighting.label -text "Lighting" -fg purple -width 15
    pack .top.lightshade.lighting.label
    foreach light { 1 2 3 4 5 6 } {
	checkbutton .top.lightshade.lighting.light$light -text "Light $light" -variable lights($light) -command "setLight $light"
	pack .top.lightshade.lighting.light$light -side top
    }
    set lights(1) 1
    button .top.lightshade.lighting.lighting -text "On" -command "toggleLighting off"
    pack .top.lightshade.lighting.lighting -side top -expand yes -fill x

    # Shading
    frame .top.lightshade.shading -relief ridge -borderwidth 3
    pack .top.lightshade.shading -side top
    label .top.lightshade.shading.label -text "Shading" -fg purple
    pack .top.lightshade.shading.label 
    foreach shade { flat gouraud wireframe } {
	radiobutton .top.lightshade.shading.shade$shade -text "$shade" -variable shadingType -value $shade -command changeShading
	pack .top.lightshade.shading.shade$shade -side top
    }
    set shadingType gouraud
    checkbutton .top.lightshade.shading.fog -text "Fog" -variable fog -command setFog 
    pack .top.lightshade.shading.fog -side top -expand yes
    
    # Annotation
    button .top.lightshade.annotation -text "Annotation" -command showAnnotateBar
    pack .top.lightshade.annotation -side top -expand yes

    # Annotation/Geom info
    button .top.lightshade.info -text "Info" -command showInfo
    pack .top.lightshade.info -side top -expand yes

    # Home button - pack later
    button .top.lightshade.gohome -text "Home" -command "ui gohome"
    
    # ZTex button - pack later
    button .top.lightshade.getztex -text "Get ZTex" -command "ui getZTex"

    ######### Bottom frame ##########
    frame .bottom -borderwidth 5
    pack .bottom -side bottom -fill x

    # Top - stats
    frame .bottom.stats 
    pack .bottom.stats -side top -fill x
    label .bottom.stats.fps -text "FPS: " -fg purple -width 30
    pack .bottom.stats.fps -side left
    label .bottom.stats.rendermode -text "Render Mode: " -fg purple -width 30
    pack .bottom.stats.rendermode -side left
    label .bottom.stats.compression -text "Compression: " -fg purple -width 30
    pack .bottom.stats.compression -side left
    label .bottom.stats.transfermode -text "Transfer Mode: PTP (default)" -fg purple -width 30
    pack .bottom.stats.transfermode -side left
    label .bottom.stats.connection -text "Not connected" -fg purple -width 30
    pack .bottom.stats.connection -side left
    
    
    # Bottom - chat
    frame .bottom.chatframe1 -borderwidth 5
    pack .bottom.chatframe1 -side top -fill x
    Scrolled_Listbox .bottom.chatframe1.chat -bg white -borderwidth 5
    pack .bottom.chatframe1.chat -side top -fill x
    frame .bottom.chatframe2 -borderwidth 5
    pack .bottom.chatframe2 -side top -fill x
    entry .bottom.chatframe2.chatentry -textvariable chatt(thetext) -bg white -relief flat -borderwidth 5 
    pack .bottom.chatframe2.chatentry -side top -fill x
    bind .bottom.chatframe2.chatentry <Return> dochat

    # Create visual list for OpenGL drawing area
    set visuals [ui listvisuals .]
    
    # Set the appropriate visual
    tkwait visibility .
    ui setvisual .top.viewframe.viewwindow 0 640 512
    pack .top.viewframe.viewwindow
    bindEvents .top.viewframe.viewwindow
}

set port_spacing 18
set port_width 13
set port_height 7

proc computeOPortCoords {modid which} {
    global port_spacing
    global port_width
    set px [expr $which*$port_spacing+6+$port_width/2]
    set at [.remote.viewerFrame.canvas coords $modid]
    set mx [lindex $at 0]
    set my [lindex $at 1]
    
    set h [winfo height .remote.viewerFrame.canvas$modid]
    
    set x [expr $px+$mx]
    set y [expr $my+$h]
    return [list $x $y]
}

proc computeIPortCoords {modid which} {
    global port_spacing
    global port_width
    set px [expr $which*$port_spacing+6+$port_width/2]
    puts "MODID = $modid"
    set at [.remote.viewerFrame.canvas coords $modid]
    
    set mx [lindex $at 0]
    set my [lindex $at 1]
        
    set x [expr $px+$mx]
`    set y $my
    return [list $x $y]
}

proc routeConnection {omodid owhich imodid iwhich} {
    set outpos [computeOPortCoords $omodid $owhich]
    set inpos [computeIPortCoords $imodid $iwhich]
    set ox [lindex $outpos 0]
    set oy [lindex $outpos 1]
    set ix [lindex $inpos 0]
    set iy [lindex $inpos 1]

    set minextend 10
    if {$ox == $ix} {
	return [list $ox $oy $ix $iy]
    } elseif {[expr $oy+2*$minextend] < $iy} {
	set my [expr ($oy+$iy)/2]
	return [list $ox $oy $ox $my $ix $my $ix $iy]
    } else {
	set mx $ox
	if {$ix < $mx} {
	    set mx $ix
	}
	return [list $ox $oy $ox [expr $oy+10] [expr $mx-50] [expr $oy+10] \
		[expr $mx-50] [expr $iy-10] $ix [expr $iy-10] $ix $iy]
    }
}

proc drawModule {modname} {
    global remoteModuleList
    global remoteCanvasWidth remoteCanvasHeight
    
    set module $remoteModuleList($modname)
    set x [lindex $module 0]
    set y [lindex $module 1]
    set connections [lindex $module 2]

    # Draw the frame
    set modframe .remote.viewerFrame.canvas$modname
    frame $modframe -relief raised -borderwidth 3
    frame $modframe.ff
    pack $modframe.ff -side top -expand yes -fill both -padx 5 -pady 6
	 
    set p $modframe.ff

    # Make the title
    label $p.title -text $modname -anchor w
    pack $p.title -side top -padx 2 -anchor w

    # Stick it in the canvas
    .remote.viewerFrame.canvas create window $x $y -window $modframe \
	    -tags $modname -anchor nw

    # Create binding
    bind $modframe <ButtonRelease-1> "ui request $modname"
    
}

global modConnections

proc rebuildModuleConnections {} {

    global modConnections
    global inports
    global remoteModuleList

    # Delete all current connections
    if { [info exists modConnections] } {
	foreach conn $modConnections {
	    .remote.viewerFrame.canvas delete $conn
	}
	set modConnections {}
    }
    # Clear all current inport counts, etc
    foreach module [array names remoteModuleList] {
	set inports($module) 0
    }

    # Rebuild the list from scratch
    puts "Names: [array names remoteModuleList]"
    foreach module [array names remoteModuleList] {
	puts "Connecting module $module"
	connectModule $module [lindex [lrange $remoteModuleList($module) 2 end] 0 ] 
	
    }
}

proc connectModule {modname connections} {
    
    # Configure ports if needed
    puts [llength $connections]
    puts "-[lindex $connections 0]-"
    if {[llength $connections] > 0 && [lindex $connections 0] != "" } {
	global inports
	
	set owhich 1
	foreach conn $connections {
	    # Our outport to the connection's inport
	    if { ![info exists inports($conn)] } {
		set inports($conn) 0
		puts "setting inports $conn to 0. Result is now $inports($conn)"
	    }
	    set path [routeConnection $modname $owhich $conn $inports($conn)]
	    puts "Path = $path"
	    lappend modConnections [.remote.viewerFrame.canvas create line $path -width 2]
	    incr inports($conn)
	    incr owhich
	}
    }
}

proc addModule {modname x y args} {
    global remoteModuleList

    set connections [list $args]

    # Add this module to the list
    set remoteModuleList($modname) [list $x $y $connections]
    set inports($modname) 0 ;# 0 Inputs now...

    # Draw it on the screen
    drawModule $modname
}

proc mkRemoteModuleViewer {} {
    global remoteCanvasHeight remoteCanvasWidth
    toplevel .remote
    wm geometry .remote 250x250

    frame .remote.viewerFrame -relief sunken -borderwidth 3
    canvas .remote.viewerFrame.canvas  \
        -scrollregion "0 0 $remoteCanvasWidth $remoteCanvasHeight" \
	-bg #036
    
    pack .remote.viewerFrame -expand yes -fill both
    pack .remote.viewerFrame.canvas -expand yes -fill both

    bind .remote.viewerFrame.canvas <ButtonRelease-3> remoteRefresh
}

proc remoteRefresh {} {
    global remoteModuleList
    global inports

    foreach module [array names remoteModuleList] {
	destroy .remote.viewerFrame.canvas$module
    }
    array set remoteModuleList {}
    array set inports {}
    
    ui remoteModule refresh
}

proc showInfo {} {
    global annotateInfo
    global annotateItems
    set f .annotateInfo

    if {[winfo exists $f]} {
	return;
    }

    toplevel $f

    set items [ui getAnnotations]

    puts "$items"

    foreach item $items {
	frame $f.frame$item
	pack $f.frame$item -side top
	label $f.frame$item.label -text $item
	pack $f.frame$item.label -side left
	checkbutton $f.frame$item.enable -text "Enable" -variable annotateItems($item) -command "enableAnnotation $item"
	pack $f.frame$item.enable -side left
	set annotateItems($item) 1
	button $f.frame$item.remove -text "Delete" -command "deleteAnnotation $item"
	pack $f.frame$item.remove -side left
	
    }
}


proc enableAnnotation {item} {
    global annotateItems

    ui enableAnnotation $item $annotateItems($item)
}

proc deleteAnnotation {item} {
    ui deleteAnnotation $item
    pack forget .annotateInfo.frame$item
}

proc doHome {} {
    pack .top.lightshade.gohome -expand yes
}

proc doZTex {} {
    pack .top.lightshade.getztex -expand yes
}

proc getAnnotateText {} {
    global annotateText
    set f .annotateText

    if [ Dialog_Create $f "Annotation Text" -borderwidth 10 ] {
	frame $f.f1
	pack $f.f1 -side top
	label $f.f1.label -text "Text to add:"
	pack $f.f1.label -side left
	entry $f.f1.text -textvariable annotateText(text) -bg white 
	pack $f.f1.text -side right
	
	frame $f.buttons
	pack $f.buttons -side bottom
	button $f.buttons.ok -text "Ok" -command {set annotateText(ok) 1}
	pack $f.buttons.ok -side left 
	button $f.buttons.cancel -text "Cancel" -command {set annotateText(ok) 0}
	pack $f.buttons.cancel -side right
	
	bind $f.f1.text <Return> {set annotateText(ok) 1 ; break}
    }

    set $annotateText(text) ""
    set annotateText(ok) 0
    Dialog_Wait $f annotateText(ok) $f.f1.text
    Dialog_Dismiss $f
    if { $annotateText(ok) } {
	return $annotateText(text)
    }
    return ""
}

proc showAnnotateBar {} {
    if {[winfo exists .annotatebar]} {
	return;
    }
    toplevel .annotatebar -width 100

    foreach b {pointer text draw} {
	button .annotatebar.$b -image [image create photo -file $b.gif] -command "ui annotateMode $b"
	pack .annotatebar.$b -side left
    }
    button .annotatebar.off -text "Off" -command "ui annotateMode off"
    pack .annotatebar.off -side left
    
}


proc showmsg {type args} {
    tk_messageBox -title $type -type ok -message "$args" -icon $type
}

proc setCompression {mode} {
    .bottom.stats.compression configure -text "Compression: $mode" 
}

proc setTransfer {mode} {
    .bottom.stats.transfermode configure -text "Transfer Mode: $mode" 
}

proc setFog {} {
    global fog

    ui setFog $fog
}

proc changeShading {} {
    global shadingType

    ui setShading $shadingType
}

proc setLight {whichlight} {
    global lights
    
    ui setlight $whichlight $lights($whichlight)

}

proc toggleLighting {state} {
    if { $state == "off" } {
	.top.lightshade.lighting.lighting configure -text "Off" -command "toggleLighting on"
	ui setLighting off
    } else {
	.top.lightshade.lighting.lighting configure -text "On" -command "toggleLighting off"
	ui setLighting on
    }
    
}


proc setFPS {fps} {

    .bottom.stats.fps configure -text "FPS: $fps"

}

proc setConnection {type servername} {
    if { $type == "connect" } {
	.bottom.stats.connection configure  -text "Connected to $servername"
	.main_menu.file.menu entryconfigure 0 -label "Disconnect" -command "ui disconnect"
    } else {
	.bottom.stats.connection configure  -text "Not connected"
	.main_menu.file.menu entryconfigure 0 -label "Connect" -command showConnectDialog
    }
    return 0
}

proc bindEvents {w} {
    bind $w <Expose> "ui redraw"
    bind $w <Configure> "ui redraw"
    
    bind $w <ButtonPress-1> "ui mtranslate start %x %y"
    bind $w <Button1-Motion> "ui mtranslate move %x %y"
    bind $w <ButtonRelease-1> "ui mtranslate end %x %y"
    bind $w <ButtonPress-2> "ui mrotate start %x %y %t"
    bind $w <Button2-Motion> "ui mrotate move %x %y %t"
    bind $w <ButtonRelease-2> "ui mrotate end %x %y %t"
    bind $w <ButtonPress-3> "ui mscale start %x %y"
    bind $w <Button3-Motion> "ui mscale move %x %y"
    bind $w <ButtonRelease-3> "ui mscale end %x %y"
}

proc setCompressors {args} {

    # Delete preexisting
    .main_menu.compression.menu delete 1 end

    # Add new ones
    .main_menu.compression.menu add command -label "None" -underline 0 \
	    -command "ui changeCompression none"
    foreach module $args {
	.main_menu.compression.menu add command -label $module -underline 0 \
		-command "ui changeCompression $module"
    }
    
    return 0
}

proc setRenderers {args} {
    global renderers

    # Delete preexisting
    .main_menu.renderers.menu delete 1 end
    set renderers {}

    # Add new ones
    foreach renderer $args {
	regsub -all " " [string tolower $renderer] "_" saferenderer
	
	.main_menu.renderers.menu add cascade -label "$renderer" -underline 0 \
		-menu .main_menu.renderers.menu.$saferenderer
	menu .main_menu.renderers.menu.$saferenderer -tearoff 0
	lappend renderers $renderer
    }

    
    return 0
}

proc setTransferModes {args} {
     # Delete preexisting
    .main_menu.transferMode.menu delete 1 end

    # Add new ones
    # default transfer mode is PTP (point to point)
    # .main_menu.transferMode.menu add command -label "None" -underline 0 \
    #	    -command "ui changeTransfer PTP"  
    foreach mode $args {
	.main_menu.transferMode.menu add command -label $mode -underline 0 \
		-command "ui changeTransfer $mode"
    }
    
    return 0

}

proc updateClientList {type args} {
    global clients

    if {$type == "add"} {
	set f "[lindex $args 0] : [lindex $args 1] : [lindex $args 2]"
	.top.clientframe.clientlist.list insert end $f 
	set g [ .top.clientframe.clientlist.list index end ]
	set f "[lindex $args 0] : [lindex $args 1]"
	set clients($f) $g
    } elseif {$type == "sub"} {
	set f "[lindex $args 0] : [lindex $args 1]"

	set clientindex [expr $clients($f)-1]

	.top.clientframe.clientlist.list delete $clientindex
	set clients [lreplace clients($f) clients($f) ]
    } elseif {$type == "modify"} {
	set f "[lindex $args 0] : [lindex $args 1]"

	set clientindex [expr $clients($f)-1]

	set f "[lindex $args 0] : [lindex $args 1] : [lindex $args 2]"
	.top.clientframe.clientlist.list delete $clientindex
	.top.clientframe.clientlist.list insert $clientindex $f
    } elseif {$type == "fill"} {
	puts "Filling clients. Args = $args\n"
	.top.clientframe.clientlist.list delete 0 end

	foreach {client addr group} $args {
	    set f "$client : $addr : $group"
	    puts $f
	    .top.clientframe.clientlist.list insert end $f
	}
    } else {
	puts "Unknown argument to updateClientList: $type\n"
	return 1
    }
    return 0
}

proc updateGroupViewer {type args} {
    global renderers

    if {$type == "add"} {
	puts "Adding group viewers. Args = $args\n"
	foreach {group viewer} $args {
	    foreach renderer $renderers {
		set foo [string first "$renderer" "$group"]
		if { $foo != -1 } {
		    regsub -all " " [string tolower $renderer] "_" saferenderer
		    .main_menu.renderers.menu.$saferenderer add command -label "$group -> $viewer" -underline 0 -command "ui changeGroup \"$group\" \"$viewer\" \"$renderer\""
		    break
		}
	    }
	}

    } elseif {$type == "sub"} {
	puts "Subbing group viewers. Args = $args\n"
    } elseif {$type == "fill"} {
	puts "Filling group viewers. Args = $args\n"
	set viewers {}

	foreach {group viewer} $args {
	    if { [lsearch -exact $viewers $viewer] == -1 } {
		lappend viewers $viewer
	    }

	    foreach renderer $renderers {
		set foo [string first "$renderer" "$group"]
		if { $foo != -1 } {
		    regsub -all " " [string tolower $renderer] "_" saferenderer
		    .main_menu.renderers.menu.$saferenderer add command -label "$group -> $viewer" -underline 0 -command "ui changeGroup \"$group\" \"$viewer\" \"$renderer\""
		    break
		}
	    }
	}

	# Add new and standalone entries
	foreach renderer $renderers {
	    foreach viewer $viewers {
		regsub -all " " [string tolower $renderer] "_" saferenderer
		.main_menu.renderers.menu.$saferenderer add command -label "New -> $viewer" -underline 0 -command "ui changeGroup New \"$viewer\" \"$renderer\""
		.main_menu.renderers.menu.$saferenderer add command -label "Standalone -> $viewer" -underline 0 -command "ui changeGroup Standalone \"$viewer\" \"$renderer\""
	    }
	}
    } else {
	puts "Unknown argument to updateGroupViewer: $type\n";
	return 1
    }
    return 0
}

proc setRendererName {name} {
    set result "Render Mode: $name"
    .bottom.stats.rendermode configure  -text "$result"
    return 0
}

proc dochat {} {
    global chatt
    ui addchat $chatt(thetext)
    addchat "Local" $chatt(thetext)
    set chatt(thetext) ""
    return 0
}

proc addchat {client chattext} {
    .bottom.chatframe1.chat.list insert end "$client: $chattext"
    .bottom.chatframe1.chat.list see end
    return 0
}

proc showConnectDialog {} {
    global connect 
    set f .connect
    
    if [ Dialog_Create $f "Connect To Server" -borderwidth 10 ] {

	foreach line { nickname server port } {
	    frame .connect.$line -width 20
	    pack .connect.$line -side top -fill x
	    label .connect.$line.label -text "$line"
	    entry .connect.$line.text -textvariable connect($line) -bg white
	    pack .connect.$line.label -side left
	    pack .connect.$line.text -side right
	}
	set connect($line) "6210"
	# Remove following line!
	set connect(server) "localhost" 
	frame .connect.buttons
	pack .connect.buttons -side bottom
	button .connect.ok -text "Ok" -command {set connect(ok) 1}
	pack .connect.ok -side left 
	button .connect.cancel -text "Cancel" -command {set connect(ok) 0}
	pack .connect.cancel -side right

	bind $f.server.text <Return> {set connect(ok) 1 ; break}
    }
    set connect(ok) 0
    Dialog_Wait $f connect(ok) $f.server.text
    Dialog_Dismiss $f
    if { $connect(ok) } {
	ui connect $connect(nickname) $connect(server) $connect(port)
    }
    return 0
}

proc Scroll_Set {scrollbar geoCmd offset size} {
    if {$offset != 0.0 || $size != 1.0} {
	eval $geoCmd
    }
    $scrollbar set $offset $size
}

proc Scrolled_Listbox {f args} { 
    frame $f
    listbox $f.list -xscrollcommand [list Scroll_Set $f.xscroll \
	    [list grid $f.xscroll -row 1 -column 0 -sticky we]] \
    -yscrollcommand [list Scroll_Set $f.yscroll \
            [list grid $f.yscroll -row 0 -column 1 -sticky ns]]
    eval {$f.list configure} $args
    scrollbar $f.xscroll -orient horizontal -command [list $f.list xview]
    scrollbar $f.yscroll -orient vertical -command [list $f.list yview]
    grid $f.list -sticky news
    grid rowconfigure $f 0 -weight 1
    grid columnconfigure $f 0 -weight 1
    return $f.list
}

proc Dialog_Create {top title args} {
    global dialog
    if [winfo exists $top] {
	switch -- [wm state $top] {
	    normal {
		raise $top 
	    }
	    withdrawn -
	    iconified { 
		wm deiconify $top
		catch { wm geometry $top $dialog(geo, $top)}
	    }
	}
	return 0
    } else { 
	eval {toplevel $top} $args
	wm title $top $title
	return 1
    }
}

proc Dialog_Wait { top varName { focus{} } } {
    upvar $varName var

    bind $top <Destroy> {list set $varName $var}
    
    set focus $top

    set old [focus -displayof $top]
    focus $focus 
    catch {tkwait visibility $top}
    catch {grab $top}

    tkwait variable $varName
    catch {grab release $top}
    focus $old
}

proc Dialog_Dismiss {top} {
    global dialog
    
    catch {
	set dialog(geo,$top) [wm geometry $top]
	wm withdraw $top
    }
}
