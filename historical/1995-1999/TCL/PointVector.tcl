 
proc makePoint {w title name command} {
    frame $w -relief groove -borderwidth 2
    label $w.label -text $title
    pack $w.label -side top
    scale $w.x -orient horizontal -variable $name-x \
	    -from -150 -to 150 -label "X:" \
	    -showvalue true -tickinterval 150 \
	    -resolution 0 -digits 5 \
	    -command $command
    pack $w.x -side top -expand yes -fill x
    entry $w.ex -textvariable $name-x
    pack $w.ex -side top -expand yes -fill x
    bind $w.ex <Return> "$command $name-x"
    scale $w.y -orient horizontal -variable $name-y \
	    -from -150 -to 150 -label "Y:" \
	    -showvalue true -tickinterval 150 \
	    -resolution 0 -digits 5 \
	    -command $command
    pack $w.y -side top -expand yes -fill x
    entry $w.ey -textvariable $name-y
    pack $w.ey -side top -expand yes -fill x
    bind $w.ey <Return> "$command $name-y"
    scale $w.z -orient horizontal -variable $name-z \
	    -from -150 -to 150 -label "Z:" \
	    -showvalue true -tickinterval 150\
	    -resolution 0 -digits 5 \
	    -command $command
    pack $w.z -side top -expand yes -fill x
    entry $w.ez -textvariable $name-z
    pack $w.ez -side top -expand yes -fill x
    bind $w.ez <Return> "$command $name-z"
}

 
proc makePlane {w title name command} {
    frame $w -relief groove -borderwidth 2
    label $w.label -text $title
    pack $w.label -side top
    scale $w.x -orient horizontal -variable $name-x \
	    -from -1 -to 1 -label "X:" \
	    -showvalue true -tickinterval 5 \
	    -resolution 0 -digits 3 \
	    -command $command
    pack $w.x -side top -expand yes -fill x
    scale $w.y -orient horizontal -variable $name-y \
	    -from -1 -to 1 -label "Y:" \
	    -showvalue true -tickinterval 5 \
	    -resolution 0 -digits 3 \
	    -command $command
    pack $w.y -side top -expand yes -fill x
    scale $w.z -orient horizontal -variable $name-z \
	    -from -1 -to 1 -label "Z:" \
	    -showvalue true -tickinterval 5 \
	    -resolution 0 -digits 3 \
	    -command $command
    pack $w.z -side top -expand yes -fill x
    expscale $w.d -orient horizontal -variable $name-d \
	    -label "D:" \
	    -command $command
#    scale $w.d -orient horizontal -variable $name-d \
#	    -from -10 -to 10 -label "D:" \
#	    -showvalue true -tickinterval 10 \
#	    -resolution 0.001 -digits 5 \
#	    -command $command
    pack $w.d -side top -expand yes -fill x
}

proc updateNormalVector {xname yname zname name1 name2 op} {
    global $xname $yname $zname
    global unv_update
    if {$unv_update} {
	puts "skipping $xname"
	return;
    }
    set $xname $value
    set unv_update 1
    set x [set $xname]
    set y [set $yname]
    set z [set $zname]
    puts "x is $x"
    puts "y is $y"
    puts "z is $z"
    if {$y == 0 && $z == 0} {
	set n [expr sqrt((1-$x*$x)/2)]
	set $yname $n
	set $zname $n
    } else {
	set a [expr sqrt((1-$x*$x)/($y*$y+$z*$z))]
	set $yname [expr $y*$a]
	set $zname [expr $z*$a]
    }
    set unv_update 0
    eval $command
}

proc makeNormalVector {w title name command} {
    frame $w -relief groove -borderwidth 2
    label $w.label -text $title
    pack $w.label -side top
    global $name-x $name-y $name-z
    global unv_update
    set unv_update 0
    trace variable $name-x w "updateNormalVector $name-x $name-y $name-z"
    trace variable $name-y w "updateNormalVector $name-y $name-z $name-x"
    trace variable $name-z w "updateNormalVector $name-z $name-x $name-y"
    scale $w.x -orient horizontal -variable $name-x \
	    -from -1 -to 1 -label "X:" \
	    -showvalue true -tickinterval 1 \
	    -resolution 0 -digits 3 \
	    -command $command
    pack $w.x -side top -expand yes -fill x
    entry $w.ex -textvariable $name-x
    pack $w.ex -side top -expand yes -fill x
    bind $w.ex <Return> "$command $name-x"    
    scale $w.y -orient horizontal -variable $name-y \
	    -from -1 -to 1 -label "Y:" \
	    -showvalue true -tickinterval 1 \
	    -resolution 0 -digits 3 \
	    -command $command
    pack $w.y -side top -expand yes -fill x
    entry $w.ey -textvariable $name-y
    pack $w.ey -side top -expand yes -fill x
    bind $w.ey <Return> "$command $name-y"    
    scale $w.z -orient horizontal -variable $name-z \
	    -from -1 -to 1 -label "Z:" \
	    -showvalue true -tickinterval 1 \
	    -resolution 0 -digits 3 \
	    -command $command
    pack $w.z -side top -expand yes -fill x
    entry $w.ez -textvariable $name-z
    pack $w.ez -side top -expand yes -fill x
    bind $w.ez <Return> "$command $name-z"    
}
