
#
#  VolVis.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   April 1996
#
#  Copyright (C) 1996 SCI Group
#

itcl_class VolVis {
    inherit Module
    constructor {config} {
	set name VolVis
	set_defaults
	global expose_number
	set expose_number 0
    }
    
    
    method set_defaults {} {
	$this-c needexecute

	global $this-xpos $this-ypos
	set $this-xpos 0.0
	set $this-ypos 0.0

	puts "the default $this-xpos"
    }
    
    method raiseGL {} {

	global expose_number
	
	set w .ui$this
	if {[winfo exists $w.gl]} {
	    raise $w.gl
	} else {
	    toplevel $w.gl
	    wm geometry $w.gl =600x600+300-200
	    wm minsize $w.gl 200 200
	    wm maxsize $w.gl 600 600

	    opengl $w.gl.gl -geometry 600x600 -doublebuffer false -direct false -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0
	    
	    bind $w.gl.gl <Expose> \
		    "if {$expose_number == 0} \
		    { set expose_number 1;$this-c redraw_all} \
		    else { puts nonzero }"
	    pack $w.gl.gl -fill both -expand 1
	}
    }
	
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w
	
	frame $w.f
	button $w.f.b -text "Redraw" -command "$this-c redraw_all"
	button $w.f.viewstuff -text "View" -command "$this makeViewPopup"
	button $w.f.rastersize -text "Raster" -command "$this adjustRasterSize"
	button $w.f.background -text "Background Color" -command "$this changeBackground"
	button $w.f.execbutton -text "Execute" -command "$this-c wanna_exec"

	button $w.f.graph -text "Opacity Map" -command "$this opacity_scalarval"

	button $w.f.tf -text "Transfer Function" -command "$this transfer_func"
	
	pack $w.f.b $w.f.viewstuff $w.f.rastersize $w.f.background \
		$w.f.graph $w.f.tf  $w.f.execbutton \
		-expand yes -fill x -pady 2 -padx 2
	
	pack $w.f
	
	raiseGL
    }
    

    method makeViewPopup {} {
	
	set w .view$this

	if {[winfo exists $w]} {
	    raise $w
	} else {

	toplevel $w
	wm title $w "View"
	wm iconname $w view
	wm minsize $w 100 100
	set c "$this-c redraw_all"
	set view $this-interactiveView
	
	makePoint $w.eyep "Eye Point" $view-eyep ""
	pack $w.eyep -side left -expand yes -fill x
	
	makePoint $w.lookat "Look at Point" $view-lookat ""
	pack $w.lookat -side left -expand yes -fill x
	
	makeNormalVector $w.up "Up Vector" $view-up ""
	pack $w.up -side left -expand yes -fill x
	
	frame $w.f -relief groove -borderwidth 2
	pack $w.f
	scale $w.f.fov -orient horizontal -variable $view-fov \
		-from 0 -to 180 -label "Field of View:" \
		-showvalue true -tickinterval 60 \
		-digits 3 

	pack $w.f.fov -expand yes -fill x

	}
    }

    method adjustRasterSize {} {
	
	set w .adjustRS$this

	if {[winfo exists $w]} {
	    raise $w
	} else {
	    toplevel $w
	    wm title $w "Raster Size"
	    wm minsize $w 200 180
	    wm maxsize $w 200 180

	    frame $w.f -relief groove -borderwidth 2
	    pack $w.f

	    scale $w.f.x -orient horizontal -variable $this-rasterX \
		-from 100 -to 600 -label "Horizontal:" \
		-showvalue true -tickinterval 150 \
		-digits 3 -length 5c
	    
	    scale $w.f.y -orient horizontal -variable $this-rasterY \
		-from 100 -to 600 -label "Vertical:" \
		-showvalue true -tickinterval 100 \
		-digits 3 -length 5c

	    pack $w.f.x $w.f.y -expand yes -fill x

	}
    }
    
    method changeBackground {} {
	
	set w .changeBackground$this

	if {[winfo exists $w]} {
	    raise $w
	} else {
	    
	    toplevel $w
	    wm title $w "Background Color"
	    wm minsize $w 134 250
	    wm maxsize $w 134 250

	    frame $w.f -relief groove -borderwidth 2
	    pack $w.f

	    scale $w.f.red -orient horizontal -variable $this-red \
		-from 0 -to 255 -label "Red" \
		-showvalue true -tickinterval 100 \
		-digits 3 -length 120
	    
	    scale $w.f.green -orient horizontal -variable $this-green \
		-from 0 -to 255 -label "Green" \
		-showvalue true -tickinterval 100 \
		-digits 3 -length 120
	    
	    scale $w.f.blue -orient horizontal -variable $this-blue \
		-from 0 -to 255 -label "Blue" \
		-showvalue true -tickinterval 100 \
		-digits 3 -length 120
	    
	    pack $w.f.red $w.f.green $w.f.blue -expand yes -fill x

	}
    }

    method opacity_scalarval {} {

	set mn .opacity_scalarval$this
	
	if {[winfo exists $mn]} {
	    raise $mn
	} else {

	    toplevel $mn -relief sunken -borderwidth 4
	    wm title $mn "Transfer function"

	    frame $mn.tog
	    set w $mn.tog

	    frame $mn.oth
	    set v $mn.oth
	    
###############################################################
########## the main frame itself

#toplevel .graph -relief sunken -borderwidth 4

#frame .graph.tog
#set w .graph.tog

#frame .graph.oth
#set v .graph.oth

set WIDTH 205
#set HEIGHT 16
set HEIGHT 10

#create main canvases: the display window
#                      the side ruler     
#                      the bottom ruler     
#                      the unimportant fillin triangle

canvas $w.display -width $WIDTH -height $WIDTH -bg grey90
canvas $w.bottom_ruler -width $WIDTH -height $HEIGHT
canvas $v.side_ruler -width $HEIGHT -height $WIDTH
canvas $v.fillin -width $HEIGHT -height $HEIGHT


$w.bottom_ruler create line 2 2 [expr $WIDTH - 2] 2

for { set i 0 } { $i <= 10 } { incr i } {
    set inter [ expr $i * 20 ]
    set x [ expr $inter + 2 ]
    $w.bottom_ruler create line $x 2 $x 10
}

$v.side_ruler create line [expr $HEIGHT - 2] 2 [expr $HEIGHT - 2] [expr $WIDTH - 2]

for { set i 0 } { $i <= 10 } { incr i } {
    set inter [ expr $i * 20 ]
    set x [ expr $inter + 2 ]
    $v.side_ruler create line [expr $HEIGHT - 8] $x [expr $HEIGHT - 1] $x
}


########### display the node position

frame $mn.mouse
set z $mn.mouse

label $z.scalar -font 6x12 -text "Scalar Val: "
label $z.opacity -font 6x12 -text "Opacity: "

#global $this-xpos $this-ypos

label $z.posX -font 6x12 -textvariable $this-xpos
label $z.posY -font 6x12 -textvariable $this-ypos

bind .opacity_scalarval$this.tog.display <Any-Motion> \
	"$this-c incredible %x %y $WIDTH"
#	"$this reportCoords %x %y $WIDTH; $this-c incredible; puts helllish"

#### packincredible

pack $z.scalar $z.posX $z.opacity $z.posY -fill x -side left

###### add a button at the bottom


button .opacity_scalarval$this.getdata \
	-text "GetData" -command "$this get_data ; $this-c get_data"

########### add some description of opacity ruler

frame $mn.leftadd


frame $mn.leftadd.rd
set u1 $mn.leftadd.rd

label $u1.joy -font 6x12 -text "0"
label $u1.top -font 6x12 -text "1"
label $u1.mid -font 6x12 -text ".5"

pack $u1.top
pack $u1.mid -pady 80
pack $u1.joy

frame $mn.leftadd.op

set u2 $mn.leftadd.op

label $u2.one -font 6x12 -text "O"
label $u2.two -font 6x12 -text "P"
label $u2.three -font 6x12 -text "A"
label $u2.four -font 6x12 -text "C"
label $u2.five -font 6x12 -text "I"
label $u2.six -font 6x12 -text "T"
label $u2.seven -font 6x12 -text "Y"

pack $u2.one $u2.two $u2.three $u2.four $u2.five $u2.six $u2.seven -side top \
	-pady 5

pack $u1 $u2 -side right

set u $mn.leftadd

########### add some description of value ruler

frame $mn.bottomadd

set t $mn.bottomadd

#label $t.min -font 6x12 -text "min"
#label $t.mid -font 6x12 -text "SCALAR VALUE"
#label $t.max -font 6x12 -text "max"

global $this-minSV $this-maxSV

label $t.mid -font 6x12 -text "SCALAR VALUE"
label $t.min -font 6x12 -textvariable $this-minSV
label $t.max -font 6x12 -textvariable $this-maxSV

#pack $t.mid -padx 60 -side left
#pack $t.max -side right
#pack $t.min -side left

pack $t.min -side left
pack $t.mid -padx 60 -side left
pack $t.max -side left

############ pack most important features

pack $z
pack $t


pack $w.display $w.bottom_ruler -side top
pack $v.side_ruler $v.fillin -side top

pack $w $v $u  -side right
pack $t -side bottom
pack $z -side bottom
pack .opacity_scalarval$this.getdata -side bottom -expand yes -fill x

############ do that graph thing...

set w $mn.tog.display

$w bind node <Any-Enter> "$this fillBlack %x %y"

$w bind node <Any-Leave> "$this fillWhite"

$mn.tog.display bind node <Button-2> "$this setCoords %x %y  "

$mn.tog.display bind node <B2-Motion> "$this callMoveNode %x %y"

focus $mn

mkNode 0 202   $w
mkNode 40 202  $w
mkNode 55 150  $w
mkNode 70 202  $w
mkNode 202 202 $w
makeEdge 1 2   $w
makeEdge 2 3   $w
makeEdge 3 4   $w
makeEdge 4 5   $w

# bind $w <Button-3> {
#     puts "$nodeX(1) $nodeY(1)"
#     puts "$nodeX(2) $nodeY(2)"
#     puts "$nodeX(3) $nodeY(3)"
#     puts "$nodeX(4) $nodeY(4)"
# }


###############################################################

	    
}

}

method mkNode { x y joyous } {
    global nodeX nodeY edgeFirst edgeSecond
    set new [ $joyous create oval [expr $x-10] [expr $y-10] \
	    [expr $x+10] [expr $y+10] -outline black \
	    -fill white -tags node]

    set nodeX($new) $x
    set nodeY($new) $y
    set edgeFirst($new) {}
    set edgeSecond($new) {}

}

method makeEdge {first second joyous } {
    global nodeX nodeY edgeFirst edgeSecond
    set edge [$joyous create line $nodeX($first) $nodeY($first) \
	    $nodeX($second) $nodeY($second) ]

    $joyous lower $edge
    lappend edgeFirst($first) $edge
    lappend edgeSecond($second) $edge
}


method moveNode { node xDist yDist joyous } {

    global nodeX nodeY edgeFirst edgeSecond
    
    if { ($node == 1) || ($node == 5) } {
	set xDist 0
    }

    if { ([expr $nodeX($node) + $xDist] > 202) } {
	set xDist [expr 202 - $nodeX($node)]
    }
    
    if { ( [expr $nodeX($node) + $xDist] < 0 ) } {
	set xDist 0
    }

    if { ([expr $nodeY($node) + $yDist] > 202) } {
	set yDist [expr 202 - $nodeY($node)]
    }
    
    if { ( [expr $nodeY($node) + $yDist] < 0 ) } {
	set yDist 0
    }

    $joyous move $node $xDist $yDist
    incr nodeX($node) $xDist
    incr nodeY($node) $yDist

    foreach edge $edgeFirst($node) {
	$joyous coords $edge $nodeX($node) $nodeY($node) \
		[lindex [$joyous coords $edge] 2] \
		[lindex [$joyous coords $edge] 3]
    }

    foreach edge $edgeSecond($node) {
	$joyous coords $edge [lindex [$joyous coords $edge] 0] \
		[lindex [$joyous coords $edge] 1] \
		$nodeX($node) $nodeY($node)
    }
}

method fillBlack { x y } {
    .opacity_scalarval$this.tog.display itemconfigure current -fill black

    global $this-xpos

    puts $this-xpos
    puts [set $this-xpos]
}

method fillWhite {} {
    .opacity_scalarval$this.tog.display itemconfigure current -fill white
}

method setCoords { x y } {

    global $this-curX $this-curY
    
    set $this-curX $x
    set $this-curY $y
}

method callMoveNode { x y } {

    global $this-curX $this-curY

    moveNode [ .opacity_scalarval$this.tog.display find withtag current ] \
	    [expr $x - [set $this-curX]] [expr $y - [set $this-curY]] \
	    .opacity_scalarval$this.tog.display
    set $this-curX $x
    set $this-curY $y
}

method reportCoords { x y width } {
    global $this-xpos $this-ypos
    global $this-maxSV $this-minSV

    set a [set $this-maxSV]
    set b [set $this-minSV]

    puts $a
    puts $b
    
    set joy [expr $a - $b]
    
    set $this-xpos [expr $x / [expr $joy * 1.0] ]
    set $this-ypos [expr $y / [expr $width-1.0] ]

    puts [set $this-xpos]
    puts [set $this-ypos]
}

method get_data { } {

    global nodeX nodeY

    global $this-n1x $this-n1y
    global $this-n2x $this-n2y
    global $this-n3x $this-n3y
    global $this-n4x $this-n4y
    global $this-n5x $this-n5y

    set $this-n1x $nodeX(1)
    set $this-n1y $nodeY(1)

    set $this-n2x $nodeX(2)
    set $this-n2y $nodeY(2)

    set $this-n3x $nodeX(3)
    set $this-n3y $nodeY(3)
    
    set $this-n4x $nodeX(4)
    set $this-n4y $nodeY(4)

    set $this-n5x $nodeX(5)
    set $this-n5y $nodeY(5)

}

method transfer_func {} {

    global nodeX nodeY
    
set w .transfer_func$this

	if {[winfo exists $w]} {
	    raise $w
	} else {
	    
	    toplevel $w
	    wm title $w "Transfer Function"
	    wm minsize $w 134 250
	    wm maxsize $w 134 250

	    frame $w.f -relief groove -borderwidth 2
	    pack $w.f

	    scale $w.f.t1x -orient horizontal -variable $this-t1x \
		-from 0 -to 10 -label "NodeX1" \
		-showvalue true -tickinterval 2 \
		-digits 2 -length 120
	    
	    scale $w.f.t1x -orient horizontal -variable $this-t1y \
		-from 0 -to 1 -label "NodeY1" \
		-showvalue true -tickinterval 0.2 \
		-digits 2 -length 120
	    
# 	      scale $w.f.green -orient horizontal -variable $this-green \
# 		  -from 0 -to 255 -label "Green" \
# 		  -showvalue true -tickinterval 100 \
# 		  -digits 3 -length 120
# 	      
# 	      scale $w.f.blue -orient horizontal -variable $this-blue \
# 		  -from 0 -to 255 -label "Blue" \
# 		  -showvalue true -tickinterval 100 \
# 		  -digits 3 -length 120
#

button $w.f.joyous -text "Get Data" -command "puts d; puts d; puts $nodeY(1) ; set $this-t1x $nodeY(1); $this-c hope"

	    pack $w.f.t1x $w.f.joyous  -expand yes -fill x
pack $w

	}
    }


}