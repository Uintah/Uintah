
#
#  VolVis.tcl
#
#  Written by:
#   Aleksandra Kuswik
#   Department of Computer Science
#   University of Utah
#   April 1996
#
#  Copyright (C) 1996 SCI Group
#

#
#
#
################################################################
#
#
#
################################################################


itcl_class VolVis {
    
    inherit Module


    #
    #
    #
    ################################################################
    #
    # constructs the VolVis class.  called when VolVis is
    # instantiated.
    #
    ################################################################
    
    constructor {config} {
	set name VolVis
	set_defaults
    }
    
    
    #
    #
    #
    ################################################################
    #
    # initialize variables.
    #
    ################################################################
    
    method set_defaults {} {

	global CanvasWidth CanvasHeight

	global $this-minSV $this-maxSV
	global $this-project $this-processors

	global $this-intervalCount
	global $this-uiopen

	################### new variables

	# if Salmon is connected to VolVis, then the user has
	# a choice to:
	#    * not use any of the data provided by Salmon
	#    * use just the view information
	#    * use all the information: view, z,and rgb values
	#
	global $this-salmon

	# allows the user to select the method
	#
	global $this-method

	# allows the user to specify the step size for Levoy's
	# method
	#
	global $this-stepsize

	global Selected

	# set variables shared by c++ and tcl

	set $this-maxSV 120
	set $this-minSV 0

	set $this-project 1
        set $this-processors 0

	# set protected variables and globals

	set AllNodeIndexes(0) {}
	set Xvalues(0)     {}
	set Yvalues(0)     {}

	set AllNodeIndexes(1) {}
	set Xvalues(1)     {}
	set Yvalues(1)     {}

	set AllNodeIndexes(2) {}
	set Xvalues(2)     {}
	set Yvalues(2)     {}

	set AllNodeIndexes(3) {}
	set Xvalues(3)     {}
	set Yvalues(3)     {}

	set redrawing       0
	set Selected        0

	set CanvasWidth  201
	set CanvasHeight 201

	set LineColor(0) black
	set LineColor(1) red
	set LineColor(2) green
	set LineColor(3) blue

	set $this-intervalCount 1

	set $this-uiopen 0
	puts "UI not open"

	set $this-salmon     0
	set $this-method     0
	set $this-stepsize   8 
    }
    
    #
    #
    #
    ################################################################
    #
    # raise the GL window or create it if not yet created.
    #
    ################################################################

    method raiseGL {} {
	
	set w .ui$this
	
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {

	    # initialize geometry and placement of the widget
	    
	    toplevel $w.gl
	    wm geometry $w.gl =600x600+300-200
	    wm minsize $w.gl 200 200
	    wm maxsize $w.gl 600 600

	    # create an OpenGL widget
	    
	    opengl $w.gl.gl -geometry 600x600 -doublebuffer false -direct true -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0

	    # every time the OpenGL widget is displayed, redraw it
	    
	    bind $w.gl.gl <Expose> "$this redraw_when_idle"
	    
	    # place the widget on the screen

	    pack $w.gl.gl -fill both -expand 1
	}
    }
    
    
    #
    #
    #
    ################################################################
    #
    # this method defines what happens when the UI button is
    # clicked on the module.  if already existant, it is raised,
    # otherwise, it is created.
    #
    ################################################################

    method ui {} {
	
	set w .ui$this

	global $this-uiopen
	
	set $this-uiopen 1
	puts "UI open is true"
	
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	
	toplevel $w
	frame $w.f

	# create a button for each function
	
	button $w.f.viewstuff -text "View" -command "$this makeViewPopup"
	button $w.f.rastersize -text "Raster" -command "$this adjustRasterSize"
	button $w.f.background -text "Background Color" -command "$this changeBackground"
	button $w.f.graph -text "Transfer Map" -command "$this transferFunction"


	frame $w.f.proj
	
	radiobutton $w.f.proj.per -text Perspective -variable $this-project \
		-value 1
	radiobutton $w.f.proj.ort -text Orthogonal  -variable $this-project \
		-value 0

	
	frame $w.f.proc

	radiobutton $w.f.proc.single -text Single   -variable $this-processors\
		-value 0
	radiobutton $w.f.proc.multi  -text Parallel -variable $this-processors\
		-value 1


	# set the salmon interaction value
	
	frame $w.f.salmon_interaction

	radiobutton $w.f.salmon_interaction.none -text None  \
		-variable $this-salmon -value 0
	radiobutton $w.f.salmon_interaction.view -text View  \
		-variable $this-salmon -value 1
	radiobutton $w.f.salmon_interaction.all -text All  \
		-variable $this-salmon -value 2


	#create the scale
	    
	frame $w.f.f -relief groove -borderwidth 2
	
	scale $w.f.f.steps -orient horizontal -variable $this-intervalCount \
		-from 1 -to 40 -label "Slices per processor: " \
		-showvalue true -tickinterval 10 \
		-digits 2 -length 5c


#	button $w.f.steps -text "StepSize" -command "$this adjustStepSize"
	
	button $w.f.b -text "Redraw" -command "$this-c redraw_all" -fg blue
	button $w.f.execbutton -text "Execute" -command "$this-c wanna_exec" \
		-fg blue

	# place the buttons in a window
#
# UNCOMMENT when orthogonal projections are introduced
#
#	pack $w.f.proj.per $w.f.proj.ort -side left -fill x

	pack $w.f.proc.single $w.f.proc.multi -side left -fill x

	pack $w.f.salmon_interaction.none $w.f.salmon_interaction.view  \
		$w.f.salmon_interaction.all -side left -fill x

	pack $w.f.f.steps -side left -fill x
	
#        pack $w.f.viewstuff $w.f.rastersize $w.f.background $w.f.proj       \
#		$w.f.allign $w.f.proc $w.f.salmon_interaction $w.f.methods  \
#		$w.f.graph  $w.f.steps $w.f.b  $w.f.execbutton              \
#		-expand yes -fill x -pady 2 -padx 2

        pack $w.f.viewstuff $w.f.rastersize $w.f.background                 \
		$w.f.proc $w.f.salmon_interaction                           \
		$w.f.graph  $w.f.f $w.f.b  $w.f.execbutton                  \
		-expand yes -fill x -pady 2 -padx 2
	pack $w.f

	# raise the OpenGL display window
	
	raiseGL

	# must bring this up because otherwise the SV-Opacity map
	# may be NULL and cause a seg fault.
	
	$this transferFunction
    }
    


    #
    #
    #
    ################################################################
    #
    # Binds certain X events to perform appropriate functions.
    # Motion on the canvas causes sliders to move along with the mouse.
    # When a mouse enters or leaves a node, its color changes.
    # When the left button is pressed while the mouse cursor is on
    # the node, one can move the node around the canvas.
    #
    ################################################################

    method CreateBindings { w } {

	global curX curY

	global Selected

	# as the mouse moves across the canvas, the sliders follow

	bind $w.main.top.gcanvas <Any-Motion> \
		"$this UpdateSliders $w %x %y; $this reportPos $w %x %y"

	# change node color to black when the mouse cursor enters

	$w.main.top.gcanvas bind node <Any-Enter> \
		"$this fillBlack $w.main.top.gcanvas"
	
	# change node color to white when the mouse cursor leaves
	
	$w.main.top.gcanvas bind node <Any-Leave> \
		"$this fillWhite $w.main.top.gcanvas"

	# memorize the initial node position 
	
	$w.main.top.gcanvas bind node <Button-1> {
	    set curX %x
	    set curY %y
	}

	#### hack

	global $this-maxSV $this-minSV

	set j [set $this-maxSV]

	# interactively move the node
	
	$w.main.top.gcanvas bind node <B1-Motion> \
		"set Selected 1; $this moveNode $w.main.top.gcanvas %x %y"

##	$w.main.top.gcanvas bind node <B1-Motion> \
##		"set Selected 1; $this moveNode $w.main.top.gcanvas %x %y; set hey %x; set xpos [expr $hey / $j ]; set ypos %y "
##

	# delete a node

	$w.main.top.gcanvas bind node <Double-Button-3>  \
		"$this deleteNode $w.main.top.gcanvas"

	# introduce a new SV-Opacity node

	bind $w.main.top.gcanvas <Double-Button-1>  \
		"$this introduceNode $w.main.top.gcanvas %x %y 0"

	# introduce a new SV-Red node

	bind $w.main.top.gcanvas <Shift-Double-Button-1>  \
		"$this introduceNode $w.main.top.gcanvas %x %y 1"

	# introduce a new SV-Green node

	bind $w.main.top.gcanvas <Control-Double-Button-1>  \
		"$this introduceNode $w.main.top.gcanvas %x %y 2"

	# introduce a new SV-Blue node

	bind $w.main.top.gcanvas <Alt-Double-Button-1>  \
		"$this introduceNode $w.main.top.gcanvas %x %y 3"

	# let go of node; no node is selected

	bind $w.main.top.gcanvas <ButtonRelease-1> { set Selected 0 }
    }


    #
    #
    #
    ################################################################
    #
    # allows the user to alter the view.
    #
    ################################################################

    method makeViewPopup {} {
	
	set w .view$this
	
	if {[winfo exists $w]} {
	    raise $w
	} else {
	    
	    # initialize variables
	    
	    toplevel $w
	    wm title $w "View"
	    wm iconname $w view
	    wm minsize $w 100 100
	    #
	    #
	    # EXP!
	    #set view $this-View
	    set view $this-eview
	    
	    # allow to adjust the eye and look at point, as well
	    # as the normal vector and the field of view angle.
	    
	    makePoint $w.eyep "Eye Point" $view-eyep ""
	    pack $w.eyep -side left -expand yes -fill x
	    
	    makePoint $w.lookat "Look at Point" $view-lookat ""
	    pack $w.lookat -side left -expand yes -fill x
	    
	    makeNormalVector $w.up "Up Vector" $view-up ""
	    pack $w.up -side left -expand yes -fill x

	    # place the points in a window
	    
	    frame $w.f -relief groove -borderwidth 2
	    scale $w.f.fov -orient horizontal -variable $view-fov \
		    -from 0 -to 180 -label "Field of View:" \
		    -showvalue true -tickinterval 90 \
		    -digits 3 

	    pack $w.f.fov -expand yes -fill x -side bottom
	    pack $w.f

	}
    }

    #
    #
    #
    ################################################################
    #
    # allows the user to adjust the step size
    #
    ################################################################

    method adjustStepSize {} {
	set w .adjustSS$this

	if { [winfo exists $w] } {
	    raise $w
	} else {
	    toplevel $w
	    wm title $w "Step Size"

	    #create the scale
	    
	    frame $w.f -relief groove -borderwidth 2

	    scale $w.f.slice -orient horizontal -variable $this-intervalCount \
		    -from 1 -to 40 -label "Slices per processor: " \
		    -showvalue true -tickinterval 10 \
		    -digits 2 -length 5c

	    # place these creations in a window
	    
#	    pack $w.f.ss $w.f.slice -expand yes -fill x
	    pack $w.f.slice -expand yes -fill x
	    pack $w.f
	}
    }
    

    #
    #
    #
    ################################################################
    #
    # allows the user to adjust the raster size.
    #
    ################################################################

    method adjustRasterSize {} {

	set w .adjustRS$this

	if {[winfo exists $w]} {
	    raise $w
	} else {

	    # initialize variables
	    
	    toplevel $w
	    wm title $w "Raster Size"

	    # create a frame with 2 scales

	    frame $w.f -relief groove -borderwidth 2

	    scale $w.f.x -orient horizontal -variable $this-eview-xres \
		    -from 100 -to 600 -label "Horizontal:" \
		    -showvalue true -tickinterval 100 \
		    -digits 3 -length 12c
	    
	    scale $w.f.y -orient horizontal -variable $this-eview-yres \
		    -from 100 -to 600 -label "Vertical:" \
		    -showvalue true -tickinterval 100 \
		    -digits 3 -length 12c

	    scale $w.f.ss -orient horizontal -variable $this-stepsize \
		    -from 1 -to 30 -label "Step Size" \
		    -showvalue true -tickinterval 5 \
		    -digits 2 -length 12c

	    # place the scales in a window
	    
	    pack $w.f.x $w.f.y $w.f.ss -expand yes -fill x
	    pack $w.f

	}
    }
    
    #
    #
    #
    ################################################################
    #
    # allows the user to change the background color
    #
    ################################################################

    method changeBackground {} {
	
	set w .changeBackground$this

	#
	#
	# EXP!!!

	if {[winfo exists $w]} {
	    raise $w
	} else {

	    # initialize variables
	    
	    toplevel $w
	    wm title $w "Background Color"
	    wm minsize $w 164 240
	    wm maxsize $w 164 240

	    # create 3 scales for RGB color

	    frame $w.f -relief groove -borderwidth 2

	    scale $w.f.red -orient horizontal -variable $this-eview-bg-r \
		    -from 0 -to 255 -label "Red" \
		    -showvalue true -tickinterval 100 \
		    -digits 3 -length 120
	    
	    scale $w.f.green -orient horizontal -variable $this-eview-bg-g \
		    -from 0 -to 255 -label "Green" \
		    -showvalue true -tickinterval 100 \
		    -digits 3 -length 120
	    
	    scale $w.f.blue -orient horizontal -variable $this-eview-bg-b \
		    -from 0 -to 255 -label "Blue" \
		    -showvalue true -tickinterval 100 \
		    -digits 3 -length 120

	    # place the scales in a window
	    
	    pack $w.f.red $w.f.green $w.f.blue -expand yes -fill x
	    pack $w.f

	}
    }



    #
    #
    #
    ################################################################
    #
    # Transfer function allows the user to specify the map between
    # the scalar value and opacity.
    #
    ################################################################

    method transferFunction {} {

	set w .transferFunction$this

	if {[winfo exists $w]} {
	    raise $w
	} else {
	    DrawWidget $w
	    CreateBindings $w

	    # position a few initial nodes

	    MakeSomeNodes $w.main.top.gcanvas
	}
	
    }


    
    #
    #
    #
    ################################################################
    #
    #
    #
    ################################################################

    method get_data { } {

	global $this-Xarray $this-Yarray
	global $this-Rsv $this-Rop
	global $this-Gsv $this-Gop
	global $this-Bsv $this-Bop
	
	set $this-Xarray $Xvalues(0)
	set $this-Yarray $Yvalues(0)

	set $this-Rsv $Xvalues(1)
	set $this-Rop $Yvalues(1)

	set $this-Gsv $Xvalues(2)
	set $this-Gop $Yvalues(2)

	set $this-Bsv $Xvalues(3)
	set $this-Bop $Yvalues(3)
    }


    

    #
    #
    #
    ################################################################
    #
    # Allows for interactive display of mouse cursor position by
    # moving sliders around.
    #
    ################################################################

    method UpdateSliders { w x y } {

	global Selected

	# delete the sliders

	$w.main.top.entireSideRuler.slid.slider delete all
	$w.main.bot.entireBottomRuler.slid.slider delete all
	
	
	if { ! $Selected } {

	    # draw the new side slider
	    
	    $w.main.top.entireSideRuler.slid.slider create line \
		    0 $y 6 $y
	    $w.main.top.entireSideRuler.slid.slider create line \
		    0 [expr $y + 1]  6 [expr $y + 1]
	    $w.main.top.entireSideRuler.slid.slider create line \
		    0 [expr $y - 1]  6 [expr $y - 1]

	    # draw the new bottom slider

	    $w.main.bot.entireBottomRuler.slid.slider create line \
		    $x 0 $x 6
	    $w.main.bot.entireBottomRuler.slid.slider create line \
		    [expr $x + 1] 1 [expr $x + 1] 6
	    $w.main.bot.entireBottomRuler.slid.slider create line \
		    [expr $x - 1] 1 [expr $x - 1] 6

	} else {
	    
	    set node    [ $w.main.top.gcanvas find withtag current ]
	    set line    $LineType($node)
	    set myIndex [lsearch $AllNodeIndexes($line) $node]
	    set newX    [lindex $Xvalues($line) $myIndex]
	    set newY    [lindex $Yvalues($line) $myIndex]

	    # draw the new side slider
	    
	    $w.main.top.entireSideRuler.slid.slider create line \
		    0 $newY 6 $newY
	    $w.main.top.entireSideRuler.slid.slider create line \
		    0 [expr $newY + 1]  6 [expr $newY + 1]
	    $w.main.top.entireSideRuler.slid.slider create line \
		    0 [expr $newY - 1]  6 [expr $newY - 1]

	    # draw the new bottom slider

	    $w.main.bot.entireBottomRuler.slid.slider create line \
		    $newX 0 $newX 6
	    $w.main.bot.entireBottomRuler.slid.slider create line \
		    [expr $newX + 1] 1 [expr $newX + 1] 6
	    $w.main.bot.entireBottomRuler.slid.slider create line \
		    [expr $newX - 1] 1 [expr $newX - 1] 6

	    
	}
	
    }

    


    #
    #
    #
    ################################################################
    #
    # unsets the flag, calls a redraw
    #
    ################################################################

    method redraw {} {
	set redrawing 0
	$this-c redraw_all
    }


    #
    #
    #
    ################################################################
    #
    # if a few redraws have been queued, and one has just taken
    # place, do not redraw a second time.
    #
    ################################################################

    method redraw_when_idle {} {

	if { ! $redrawing } {
	    after idle $this redraw
	    set redrawing 1
	}
    }


    #
    #
    #
    ################################################################
    #
    # the node becomes black.
    #
    ################################################################

    method fillBlack { w } {
	$w itemconfigure current -fill black
    }


    #
    #
    #
    ################################################################
    #
    # the node becomes white.
    #
    ################################################################

    method fillWhite { w } {
	$w itemconfigure current -fill white
    }


    
    #
    #
    #
    ################################################################
    #
    # creates a gridded canvas for placement of nodes
    #
    ################################################################

    method GCanvas { where width height hGrid vGrid } {

	puts "creating the canvas"

	canvas $where -width $width -height $height -bg grey90

	# draw vertical grid lines
	
	for { set i 0 } { $i <= $width } { set i [ expr $i + $hGrid ] } {
	    $where create line $i 0 $i $height
	}

	# draw horizontal grid lines

	for { set i 0 } { $i <= $height } { set i [ expr $i + $vGrid ] } {
	    $where create line 0 $i $width $i
	}

	pack $where -padx 2 -pady 2
	
    }



    #
    #
    #
    ################################################################
    #
    # Initial positioning of the nodes.
    #
    ################################################################

    method MakeSomeNodes { where } {

	puts "Making some nodes"

	makeNode $where 0 200 0
	makeNode $where 40 200 0
	makeNode $where 55 150 0
	makeNode $where 70 200 0
	makeNode $where 200 200 0

	makeNode $where 0 0 1
	makeNode $where 200 0 1

	makeNode $where 0 0 2
	makeNode $where 200 0 2

	makeNode $where 0 0 3
	makeNode $where 200 0 3

	makeEdge $where 0 1 0
	makeEdge $where 1 2 0
	makeEdge $where 2 3 0
	makeEdge $where 3 4 0

	makeEdge $where 0 1 1
	makeEdge $where 0 1 2
	makeEdge $where 0 1 3
    }


    #
    #
    #
    ################################################################
    #
    # Draws the transfer function canvas, rulers, and the fillin
    # square.
    #
    ################################################################

    method DrawWidget { w } {

	global CanvasHeight CanvasWidth

	# initialize variables
	
	toplevel $w
	wm title $w "Transfer Map"
	wm minsize $w 300 300
	wm maxsize $w 300 300

	# create a main frame for main stuff

	frame $w.main

	# create frames for 1. side ruler+canvas
	#                   2. fillin square+bottom ruler

	frame $w.main.top
	frame $w.main.bot
	frame $w.main.pos

	# create a gridded canvas inside that frame

	set HGrid         20
	set VGrid         20
	set RulerWidth    60
	
	GCanvas $w.main.top.gcanvas $CanvasWidth $CanvasHeight \
		$HGrid $VGrid

	# create the side ruler

	SideRuler $w.main.top.entireSideRuler $RulerWidth $CanvasHeight \
		$VGrid

	# create the fillin rectangle

	canvas $w.main.bot.fillin -width $RulerWidth \
		-height $RulerWidth
	# this fillin square used to be blue: -bg blue

	# create the bottom ruler

	BottomRuler $w.main.bot.entireBottomRuler \
		$CanvasWidth $RulerWidth $HGrid

	#### create the position label

	ReportPosition $w.main.pos.position $CanvasWidth 20

	pack $w.main.top.entireSideRuler $w.main.top.gcanvas \
		-side right -anchor nw

	pack $w.main.bot.entireBottomRuler $w.main.bot.fillin \
		-side right -anchor nw

	pack $w.main.pos.position -side right -anchor nw

	pack $w.main.top $w.main.bot $w.main.pos -side top
	pack $w.main
    }

    method ReportPosition { where width height } {

	global xpos ypos

	frame $where
	label $where.x -text "X = "
	label $where.y -text "Y = "

	entry $where.xx -width 6 -relief sunken -bd 2 -textvariable xpos
	entry $where.yy -width 6 -relief sunken -bd 2 -textvariable ypos

	set xpos "hi"
	set ypos "ih"
	
	pack $where.x $where.xx $where.y $where.yy -side left

    }


    method reportPos { w x y } {
	global xpos ypos
	global $this-maxSV $this-minSV
	global CanvasWidth CanvasHeight
	global Selected

	set temp [ expr [ expr [set $this-maxSV] - [set $this-minSV] ] * 1.0 ]
	
	if { $Selected } {
	    
	    set node    [ $w.main.top.gcanvas find withtag current ]
	    set line    $LineType($node)
	    set myIndex [lsearch $AllNodeIndexes($line) $node]
	    set x    [lindex $Xvalues($line) $myIndex]
	    set y    [lindex $Yvalues($line) $myIndex]
	}
	
	set xpos [ expr [set $this-minSV] +  \
		[ expr [ expr [expr $x * 1.0] / [expr $CanvasWidth - 1 ]]  \
		* $temp ] ]
	
	set ypos [expr 1 - [expr [expr $y * 1.0] / [expr $CanvasHeight - 1] ] ]

#	puts "X = $xpos, Y = $ypos"
    }

    #
    #
    #
    ################################################################
    #
    # Creates the entire bottom ruler which consists of the
    # name, ruler, and slider areas.
    #
    ################################################################

    method BottomRuler { where width height hgrid } {

	puts "creating the bottom ruler"

	set w $where; frame $w
	
	# create the frames for each of the components

	set name $w.name; frame $name

	set rule $w.ruler; frame $rule

	set slid $w.slid; frame $slid
	

	# work on the name part of the ruler

	NameBottomRuler $name [expr $width - 16]

	# work on the ruler part

	canvas $rule.canv -width [expr $width + 10] -height 10

	DrawBottomRuler $rule.canv [expr $width + 10] 10 $hgrid

	# work on the slider part
	# this is what shows the position of the mouse

	canvas $slid.slider -width [expr $width + 10] -height 6
	
	PrepareBottomSlider $slid.slider [expr $width + 10] 6

	# place frames in window

	pack $rule.canv $slid.slider
	#    pack $name $rule $slid -side bottom

	pack $slid $rule $name -side top
    }

    
    #
    #
    #
    ################################################################
    #
    # creates the name of the bottom ruler
    #
    ################################################################

    method NameBottomRuler { where width } {

	global $this-minSV $this-maxSV

	# create a frame for each label

	frame $where.a; frame $where.b; frame $where.c

# these used to be white:
#	 label $where.a.txt -font 6x12 -text "SCALAR VALUE"      -bg white
#	 label $where.b.min -font 6x12 -textvariable $this-minSV -bg white
#	 label $where.c.max -font 6x12 -textvariable $this-maxSV -bg white
#

	# create labels
	
	label $where.a.txt -font 6x12 -text "SCALAR VALUE"
	label $where.b.min -font 6x12 -textvariable $this-minSV
	label $where.c.max -font 6x12 -textvariable $this-maxSV

	# place labels in the frame

	pack $where.a.txt -padx [expr $width / 4]
	pack $where.b.min
	pack $where.c.max
	
	pack $where.b $where.a $where.c -side left -expand 0
    }


    #
    #
    #
    ################################################################
    #
    # creates the bottom ruler (line with tickmarks)
    #
    ################################################################

    method DrawBottomRuler { where width height tickinterv } {

	$where create line 1 1  [expr $width - 9] 1

	for { set i 1 } { $i <= $width } { set i [expr $i + $tickinterv] } {
	    $where create line $i 1  $i [expr $height - 2]
	}
    }



    #
    #
    #
    ################################################################
    #
    # alligns the slider at minimum scalar value
    #
    ################################################################

    method PrepareBottomSlider { where width height } {

	$where create line 1 0  1 $height
	$where create line 2 0  2 $height
	$where create line 3 0  3 $height
    }


    
    #
    #
    #
    ################################################################
    #
    # creates the "OPACITY" tag and some description for the
    # ruler part
    #
    ################################################################

    method NameSideRuler { where height } {

	set ticks $where.ticks; frame $ticks

	label $ticks.min -font 6x12 -text "0"
	label $ticks.mid -font 6x12 -text "0.5"
	label $ticks.max -font 6x12 -text "1"

	pack $ticks.max
	pack $ticks.mid -pady [expr [expr $height - 40] / 2]
	pack $ticks.min

	set op  $where.op; frame $op

	label $op.one -font 6x12 -text "O"
	label $op.two -font 6x12 -text "P"
	label $op.thr -font 6x12 -text "A"
	label $op.fou -font 6x12 -text "C"
	label $op.fiv -font 6x12 -text "I"
	label $op.six -font 6x12 -text "T"
	label $op.sev -font 6x12 -text "Y"

	pack $op.one $op.two $op.thr $op.fou $op.fiv $op.six $op.sev \
		-side top -pady 4

	pack $op $ticks -side left
    }



    #
    #
    #
    ################################################################
    #
    # draws the ruler (line with tickmarks)
    #
    ################################################################

    method DrawSideRuler { where width height tickinterv } {

	$where create line [expr $width - 2] 3 \
		[expr $width - 2] [expr $height - 8]

	for { set i 3 } { $i <= $height } { set i [expr $i + $tickinterv] } {
	    $where create line [expr $width - 8] $i [expr $width - 1] $i
	}
    }




    #
    #
    #
    ################################################################
    #
    # alligns the slider at position 0 (opacity = 0)
    #
    ################################################################

    method PrepareSideSlider { where width height } {

	$where create line 0 [expr $height - 6]  $width [expr $height - 6]
	$where create line 0 [expr $height - 7]  $width [expr $height - 7]
	$where create line 0 [expr $height - 8]  $width [expr $height - 8]
    }


    #
    #
    #
    ################################################################
    #
    # creates the entire side ruler which consists of the
    # name, ruler, and slider areas.
    #
    ################################################################

    method SideRuler { where width height tickinterv } {

	puts "creating the side ruler"

	# create a frame for the entire ruler
	
	set w $where; frame $w

	# create the frames for each of the components

	set name $w.name; frame $name

	set rule $w.ruler; frame $rule

	set slid $w.slid; frame $slid

	# work on the name part of the ruler

	NameSideRuler $name [expr $height - 6]

	# work on the ruler part

	canvas $rule.canv -width 10 -height [expr $height + 10]

	DrawSideRuler $rule.canv 10 [expr $height + 10] $tickinterv

	# work on the slider part
	# this is what shows the position of the mouse

	canvas $slid.slider -width 6 -height [expr $height + 10]
	
	PrepareSideSlider $slid.slider 6 [expr $height + 10]

	# place frames in window

	pack $rule.canv $slid.slider
	pack $name $rule $slid -side left
	
    }


    #################################################################
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    # ---------------------------------------------------------------
    #################################################################


    #
    #
    #
    ################################################################
    #
    # draws a node on the screen, and adds appropriate data to
    # the AllNodeIndexes, Xvalues, and Yvalues lists.
    #
    ################################################################

    method makeNode { joyous x y line } {

	global CanvasWidth

	# check if there exists another node positioned at x.
	# if so, move x to x+1 and make sure that there is no
	# duplicate in the x-value.  proceed until no other node
	# with position x exists.

	while { ( [lsearch $Xvalues($line) $x] != -1 ) && ( $x < $CanvasWidth ) } {
	    incr x
	}

	if { $x >= $CanvasWidth } {
	    puts "Error: cannot make node"
	} else {
	    
	    # draw a new node
	    
	    set new [ $joyous create oval [expr $x-2] [expr $y-2] \
		    [expr $x+2] [expr $y+2] -outline $LineColor($line) \
		    -fill white -tags node]

	    # initialize the edge lists corresponding to each node
	    
	    set edgeFirst($new) {}
	    set edgeSecond($new) {}

	    # reflect the addition in the lists

	    placeInLists $new $x $y $line
	}
    }


    #
    #
    #
    ################################################################
    #
    # draws an edge between two given nodes.
    #
    ################################################################

    method makeEdge { joyous one two line } {
	
	# find the global index to both the nodes

	set first  [ lindex $AllNodeIndexes($line) $one ]
	set second [ lindex $AllNodeIndexes($line) $two ]
	
	# draw the line

	set edge [$joyous create line \
		[lindex $Xvalues($line) $one] [lindex $Yvalues($line) $one] \
		[lindex $Xvalues($line) $two] [lindex $Yvalues($line) $two] \
		-fill $LineColor($line) ]


	# reflect the edge addition in the edge lists

	$joyous lower $edge
	lappend edgeFirst($first) $edge
	lappend edgeSecond($second) $edge
    }



    #
    #
    #
    ################################################################
    #
    # Interactively moves the node.
    #
    ################################################################

    method moveNode { joyous x y } {

	    global curX curY
	    global CanvasHeight

	    # associate the node with a tag
	    
	    set node [ $joyous find withtag current ]

	set line $LineType($node)

	# xDist and yDist represent how many pixels the node
	    # has moved
	    
	    set xDist [expr $x - $curX]
	    set yDist [expr $y - $curY]

	    # remember previous node position and associated index
	    # into the lists.  also, what are the new {X,Y} values?
	    
	    set myIndex [lsearch $AllNodeIndexes($line) $node]

	    set myXval  [lindex $Xvalues($line) $myIndex]
	    set myYval  [lindex $Yvalues($line) $myIndex]

	    set newXval [expr $myXval + $xDist]
	    set newYval [expr $myYval + $yDist]
	    
	    # make sure that the node positions are not registered
	    # as invalid values

	    if { ( $newYval >= $CanvasHeight) } {
		set yDist [expr [expr $CanvasHeight - $myYval] - 1]
		set newYval [expr $CanvasHeight - 1]
	    }
	    
	    if { ( $newYval < 0 ) } {
		set yDist [expr 0 - $myYval]
		set newYval 0
	    }

	    if { ( $myIndex == 0 ) || ( $myIndex == [expr [llength $AllNodeIndexes($line)] - 1] ) } {

		# the first and the last nodes must stay on the edges
		# (cannot move in the x direction)

		set xDist 0
		set newXval $myXval

	    } else {

		# make sure that one node does not go further than its
		# neighbor in the x direction.  no need to check this if
		# the node is one of the edge nodes.
		

		# know the x-positions of nodes around me
		
		set leftXval  [lindex $Xvalues($line) [expr $myIndex - 1] ]
		set rightXval [lindex $Xvalues($line) [expr $myIndex + 1] ]

		# compare the x values of nodes

		if { $leftXval >= $newXval } {
		    set newXval [expr $leftXval + 1]
		    set xDist [expr $newXval - $myXval]
		}

		if { $newXval >= $rightXval } {
		    set newXval [expr $rightXval - 1]
		    set xDist [expr $newXval - $myXval]
		}
	    }

	    # move the node on the canvas
	    
	    $joyous move $node $xDist $yDist

	    # update the lists with new x,y values

	    set Xvalues($line) [lreplace $Xvalues($line) $myIndex $myIndex \
		    $newXval]
	    set Yvalues($line) [lreplace $Yvalues($line) $myIndex $myIndex \
		    $newYval]

	    
	    # connect the node to the nodes on the left and right of it

	    foreach edge $edgeFirst($node) {
		$joyous coords $edge $newXval $newYval    \
			[lindex [$joyous coords $edge] 2] \
			[lindex [$joyous coords $edge] 3]
	    }

	    foreach edge $edgeSecond($node) {
		$joyous coords $edge [lindex [$joyous coords $edge] 0] \
			[lindex [$joyous coords $edge] 1] \
			$newXval $newYval
	    }

	    # memorize the current x,y positions
	    
	    set curX $newXval
	    set curY $newYval
    }


    #
    #
    #
    ################################################################
    #
    #
    #
    ################################################################

    method deleteNode { where } {
	
	# associate the node with a tag
	
	set node [ $where find withtag current ]

	set line $LineType($node)

	set myIndex [lsearch $AllNodeIndexes($line) $node]

	# cannot delete the edge nodes
	
	if { ( $myIndex != 0 ) &&
	( $myIndex != [expr [llength $AllNodeIndexes($line)] - 1] ) } {

#	    puts "deleting node $node ($myIndex)"

	    # deleting associated edges

	    foreach edge $edgeFirst($node) {
		$where delete $edge
	    }

	    foreach edge $edgeSecond($node) {
		$where delete $edge
	    }

	    # deleting the node itself

	    $where delete $node

	    # deleting appropriate list entries
	    
	    set AllNodeIndexes($line) [lreplace $AllNodeIndexes($line) \
		    $myIndex $myIndex]
	    set Xvalues($line)        [lreplace $Xvalues($line) \
		    $myIndex $myIndex]
	    set Yvalues($line)        [lreplace $Yvalues($line) \
		    $myIndex $myIndex]

	    # connect the other 2 edges together

	    makeEdge $where [expr $myIndex - 1] $myIndex $line
	} else {
	    puts "Warning: cannot delete an edge node"
	}
	
    }
    



    #
    #
    #
    ################################################################
    #
    # arrange the lists in order such that the xvalues are in
    # ascending order
    # the x value to be inserted will not equal any of the other
    # members of Xvalues.
    #
    ################################################################

    method placeInLists { node x y line } {

	# set the line type for the particular node

	set LineType($node) $line

	global LastNodeMadeIndex

	set len   [llength $AllNodeIndexes($line)]
	set LastNodeMadeIndex $len

	if { $len == 1 } {
	    if { $x < [lindex $Xvalues($line) 0] } {
		set LastNodeMadeIndex 0
	    }
	} else {
	    
	    for { set i 0 } { $i < [llength $AllNodeIndexes($line)] } { incr i } {
		set aaa [lindex $AllNodeIndexes($line) $i]
		set bbb [lindex $Xvalues($line) $i]
		set ccc [lindex $Yvalues($line) $i]
		
		if { ($bbb > $x)  && ($LastNodeMadeIndex == $len) } {
		    set LastNodeMadeIndex $i
		}
	    }
	}

	set AllNodeIndexes($line) [linsert $AllNodeIndexes($line) \
		$LastNodeMadeIndex $node]
	set Xvalues($line) [linsert $Xvalues($line) \
		$LastNodeMadeIndex $x]
	set Yvalues($line) [linsert $Yvalues($line) \
		$LastNodeMadeIndex $y]
    }
    




    #
    #
    #
    ################################################################
    #
    # arrange the lists in order such that the xvalues are in
    # ascending order
    # the x value to be inserted will not equal any of the other
    # members of Xvalues.
    #
    ################################################################

    method introduceNode { where x y line } {

	global LastNodeMadeIndex

	# makes a new node, attaches it to the end of ANI, Xv, Yv lists
	
	makeNode $where $x $y $line

	set node  [lindex $AllNodeIndexes($line) $LastNodeMadeIndex]
	set left  [lindex $AllNodeIndexes($line) \
		[expr $LastNodeMadeIndex - 1] ]
	set right [lindex $AllNodeIndexes($line) \
		[expr $LastNodeMadeIndex + 1] ]

	# disconnect the nodes around the new node

	foreach edge $edgeFirst($left) {
	    $where delete $edge
	}

	foreach edge $edgeSecond($right) {
	    $where delete $edge
	}

	# connect these nodes to the new node

	makeEdge $where [expr $LastNodeMadeIndex - 1] $LastNodeMadeIndex $line
	makeEdge $where $LastNodeMadeIndex [expr $LastNodeMadeIndex + 1] $line
    }

    protected redrawing
    
    protected AllNodeIndexes
    protected Xvalues
    protected Yvalues

    protected edgeFirst
    protected edgeSecond

    protected LineColor
    protected LineType
}
