 
proc makePoint {w title name command} {
    frame $w -relief groove -borderwidth 2
    label $w.label -text $title
    pack $w.label -side top
    fscale $w.x -orient horizontal -variable x,$name \
	    -from -10 -to 10 -label "X:" \
	    -showvalue true -tickinterval 5 \
	    -activeforeground SteelBlue2 -resolution .0001 -digits 3 \
	    -command $command
    pack $w.x -side top -expand yes -fill x
    fscale $w.y -orient horizontal -variable y,$name \
	    -from -10 -to 10 -label "Y:" \
	    -showvalue true -tickinterval 5 \
	    -activeforeground SteelBlue2 -resolution .0001 -digits 3 \
	    -command $command
    pack $w.y -side top -expand yes -fill x
    fscale $w.z -orient horizontal -variable z,$name \
	    -from -10 -to 10 -label "Z:" \
	    -showvalue true -tickinterval 5 \
	    -activeforeground SteelBlue2 -resolution .0001 -digits 3 \
	    -command $command
    pack $w.z -side top -expand yes -fill x
}

proc updateNormalVector {xname yname zname command value} {
    global $xname $yname $zname
    global unv_update
    if {$unv_update} {
	return;
    }
    set $xname $value
    set unv_update 1
    set x [set $xname]
    set y [set $yname]
    set z [set $zname]
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
    global x,$name y,$name z,$name
    global unv_update
    set unv_update 0
    fscale $w.x -orient horizontal -variable x,$name \
	    -from -1 -to 1 -label "X:" \
	    -showvalue true -tickinterval 1 \
	    -activeforeground SteelBlue2 -resolution .00001 -digits 3 \
	    -command "updateNormalVector x,$name y,$name z,$name \"$command\" "
    pack $w.x -side top -expand yes -fill x
    fscale $w.y -orient horizontal -variable y,$name \
	    -from -1 -to 1 -label "Y:" \
	    -showvalue true -tickinterval 1 \
	    -activeforeground SteelBlue2 -resolution .00001 -digits 3 \
	    -command "updateNormalVector y,$name z,$name x,$name \"$command\" "
    pack $w.y -side top -expand yes -fill x
    fscale $w.z -orient horizontal -variable z,$name \
	    -from -1 -to 1 -label "Z:" \
	    -showvalue true -tickinterval 1 \
	    -activeforeground SteelBlue2 -resolution .00001 -digits 3 \
	    -command "updateNormalVector z,$name x,$name y,$name \"$command\" "
    pack $w.z -side top -expand yes -fill x
}
