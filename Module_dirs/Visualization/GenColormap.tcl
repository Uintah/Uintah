#
#  GenColormap.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename GenColormap ""}

itcl_class GenColormap {
    inherit Module
    constructor {config} {
	set name GenColormap
	set_defaults
    }
    method set_defaults {} {
	global $this-nlevels
	set $this-nlevels 50
	
	global $this-interp_type
	set $this-interp_type Linear

	global $this-cinterp_type
	set $this-cinterp_type RGB

	global $this-material
	initMaterial $this-material

	$this-c setmap Rainbow
    }

    method setsize {w h config} {
	set canvasx [expr $w-1]
	set canvasy [expr $h-1]

	for {set i 0} {$i<[llength $keys]} {incr i 1} {
	    set kid [lindex $keyids $i]
	    set x [expr $canvasx*[lindex $keys $i]]
	    .ui$this.f.canvas coords $kid \
		    [expr $x-2] 0 [expr $x+2] $canvasy
	}
	$this repaint
    }

    protected canvasx 700
    protected canvasy 80

    public matwin

    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    $this update
	    return;
	}
	toplevel $w
	wm minsize $w 100 100
	frame $w.f
	set colmap $w.f

	frame $colmap.mbar -relief raised -bd 2
	pack $colmap.mbar -side top -padx 2 -pady 2 -fill x

	menubutton $colmap.mbar.type -text Defaults -menu $colmap.mbar.type.menu
	menubutton $colmap.mbar.interp -textvariable $this-interp_type \
		-menu $colmap.mbar.interp.menu
	menubutton $colmap.mbar.cinterp -textvariable $this-cinterp_type \
		-menu $colmap.mbar.cinterp.menu
	pack $colmap.mbar.type $colmap.mbar.interp $colmap.mbar.cinterp -side left
	
	menu $colmap.mbar.type.menu
	$colmap.mbar.type.menu add command -label "Rainbow" \
		-command "$this colormapWarning Rainbow"
	$colmap.mbar.type.menu add command -label "Voltage" \
		-command "$this colormapWarning Voltage"
	$colmap.mbar.type.menu add command -label "Red to Green" \
		-command "$this colormapWarning RedToGreen"
	$colmap.mbar.type.menu add command -label "Red to Blue" \
		-command "$this colormapWarning RedToBlue"
	$colmap.mbar.type.menu add command -label "Gray Scale" \
		-command "$this colormapWarning Grayscale"
	$colmap.mbar.type.menu add command -label "Spline" \
		-command "$this colormapWarning Spline"

	menu $colmap.mbar.interp.menu
	$colmap.mbar.interp.menu add radiobutton -label "Linear" \
		-variable $this-interp_type -value Linear \
		-command "$this-c interp; $this repaint"
	$colmap.mbar.interp.menu add radiobutton -label "Spline" \
		-variable $this-interp_type -value Spline \
		-command "$this splineWarning"

	menu $colmap.mbar.cinterp.menu
	$colmap.mbar.cinterp.menu add radiobutton -label "RGB" \
		-variable $this-cinterp_type -value RGB -command "$this-c cinterp; $this repaint"
	$colmap.mbar.cinterp.menu add radiobutton -label "HSV" \
		-variable $this-cinterp_type -value HSV -command "$this-c cinterp; $this repaint"

	tk_menuBar $colmap.mbar $colmap.mbar.type $colmap.mbar.interp $colmap.mbar.cinterp
	focus $colmap.mbar

	global $this-nlevels
	scale $colmap.scale -from 1 -to 500 -resolution 1 \
		-variable $this-nlevels -orient horizontal\
		-command "$this-c nlevels"
	pack $colmap.scale -side top -padx 2 -pady 2 -fill x

	canvas $colmap.canvas -scrollregion {0 0 701 81} -width 701 -height 81
	pack $colmap.canvas -side top -padx 2 -pady 2 -fill both -expand yes
	pack $colmap -fill both -expand yes

	bind $colmap.canvas <Configure> "$this setsize %w %h"
	bind $colmap.canvas <ButtonPress-2> "$this addxkey %x"

	set matwin $colmap.matwin
	global $colmap.matwin.mati

	$this update
    }

    method update {} {
	.ui$this.f.canvas delete all
	set kid0 [.ui$this.f.canvas create rectangle \
		0 0 4 $canvasy \
		-tags "keys" \
		-fill #00000000ffff \
		-outline #00000000ffff]
	.ui$this.f.canvas bind $kid0 <Double-ButtonPress-1> \
		"$this button $kid0"
	set kid1 [.ui$this.f.canvas create rectangle \
		[expr $canvasx-4] 0 $canvasx $canvasy \
		-tags "keys" \
		-fill #00000000ffff \
		-outline #00000000ffff]
	.ui$this.f.canvas bind $kid1 <Double-ButtonPress-1> \
		"$this button $kid1"

	set keyids [list $kid0 $kid1]
	
	$this repaint
    }

    method motion {kid x} {
	set kidx [$this findkey $kid]
	if [expr $kidx<0] return
	if [expr (($x >= 0) && ($x <= $canvasx))] {
	    set fx [expr $x/double($canvasx+1)]
	    if [expr ((($kidx > 0) \
		    && ($fx <= [lindex $keys [expr ($kidx)-1]])) \
		    || (($kidx < [expr [llength $keys]-1]) \
		    && ($fx >= [lindex $keys [expr ($kidx)+1]])))] {
		return
	    } else {
		set keys [lreplace $keys $kidx $kidx $fx]
		.ui$this.f.canvas coords $kid \
			[expr $x-2] 0 [expr $x+2] $canvasy
		$this-c movekey $kidx $fx
		$this repaint
	    }
	}
    }

    method button {kid} {
	set kidx [$this findkey $kid]
	if [expr $kidx<0] return
	global $matwin.mati
	set $matwin.mati $kidx

	$this-c getmat $kidx

	if {[winfo exists $matwin]} {
	    meresync $matwin
	    raise $matwin
	} else {
	    toplevel $matwin
	    makeMaterialEditor $matwin $this-material \
		    "$this commit" "$this cancel"
	}
    }

    method repaint {} {
	global $this-nlevels
	set nl [set $this-nlevels]
	.ui$this.f.canvas delete pots
	set prex [expr $canvasx/double($nl)]
	set prey [expr $canvasy/10.0]
	set prey2 [expr $prey-1]
	set prey4 [expr $prey*2-1]
	set x 0
	for {set i 0} {$i < $nl} {incr i 1} {
	    set slice [$this-c getslice $i]
	    set oldx $x
	    set x [expr ($i+1)*$prex]
	    # ambient
	    set y 0
	    set color [format #%04x%04x%04x [lindex $slice 0] [lindex $slice 1] [lindex $slice 2]]
	    .ui$this.f.canvas create rectangle \
		    $oldx $y $x [expr $y+$prey4] \
		    -tags "pots" \
		    -fill $color -outline $color
	    # diffuse
	    set y [expr $prey*2]
	    set color [format #%04x%04x%04x [lindex $slice 3] [lindex $slice 4] [lindex $slice 5]]
	    .ui$this.f.canvas create rectangle \
		    $oldx $y $x [expr $y+$prey4] \
		    -tags "pots" \
		    -fill $color -outline $color
	    # specular
	    set y [expr $prey*4]
	    set color [format #%04x%04x%04x [lindex $slice 6] [lindex $slice 7] [lindex $slice 8]]
	    .ui$this.f.canvas create rectangle \
		    $oldx $y $x [expr $y+$prey4] \
		    -tags "pots" \
		    -fill $color -outline $color
	    # emission
	    set y [expr $prey*6]
	    set color [format #%04x%04x%04x [lindex $slice 9] [lindex $slice 10] [lindex $slice 11]]
	    .ui$this.f.canvas create rectangle \
		    $oldx $y $x [expr $y+$prey4] \
		    -tags "pots" \
		    -fill $color -outline $color
	    # reflectivity
	    set y [expr $prey*8]
	    set color [format #%04x%04x%04x [lindex $slice 12] [lindex $slice 12] [lindex $slice 12]]
	    .ui$this.f.canvas create rectangle \
		    $oldx $y $x [expr $y+$prey2] \
		    -tags "pots" \
		    -fill $color -outline $color
	    # transparency
	    set y [expr $prey*9]
	    set color [format #%04x%04x%04x [lindex $slice 13] [lindex $slice 13] [lindex $slice 13]]
	    .ui$this.f.canvas create rectangle \
		    $oldx $y $x [expr $y+$prey2] \
		    -tags "pots" \
		    -fill $color -outline $color
	}
	.ui$this.f.canvas raise keys
    }

    protected keys [list 0 1]
    protected keyids ""

    method addxkey {x} {
	$this addkey [expr $x/double($canvasx)]
	$this repaint
    }

    method addkey {f} {
	set x [expr $f*$canvasx]
	set idx -1
	for {set i 0} {$i<[llength $keys]} {incr i 1} {
	    if [expr [lindex $keys $i] < $f] {
		set idx $i
	    }
	}
	incr idx 1

	set kid [.ui$this.f.canvas create rectangle \
		[expr $x-2] 0 [expr $x+2] $canvasy \
		-tags "keys midkeys" \
		-fill #ffff00000000 \
		-outline #ffff00000000]
	.ui$this.f.canvas bind $kid <Button1-Motion> \
		"$this motion $kid %x"
	.ui$this.f.canvas bind $kid <Double-ButtonPress-1> \
		"$this button $kid"
	.ui$this.f.canvas bind $kid <ButtonPress-3> \
		"$this delkey $kid"

	set keys [linsert $keys $idx $f]
	set keyids [linsert $keyids $idx $kid]
	$this-c addkey $f $idx
    }

    method delkey {kid} {
	if [expr [llength $keys] == 4] {
	    toplevel .ui$this.warn
	    tk_dialog .ui$this.warn "Insufficient Number of Keys" \
		    "You must have at least four keys for \"Spline\" \
		    interpolation." \
		    warning 0 "Ok"
	    return
	}
	
	set kidx [$this findkey $kid]
	if [expr $kidx<0] return

	.ui$this.f.canvas delete $kid

	set keys [lreplace $keys $kidx $kidx]
	set keyids [lreplace $keyids $kidx $kidx]
	$this-c delkey $kidx
	$this repaint
    }

    method clearkeys {} {
	if [expr [llength $keys] <= 2] return

	.ui$this.f.canvas delete midkeys

	set keys [lreplace $keys 1 [expr [llength $keys]-2]]
	set keyids [lreplace $keyids 1 [expr [llength $keyids]-2]]
    }

    method findkey {kid} {
	return [lsearch -exact $keyids $kid]
    }

    method commit {} {
	global $matwin.mati
	$this-c setmat [set $matwin.mati]
	set idx [set $matwin.mati]
	$this repaint

	destroy $matwin
    }
    method cancel {} {
	destroy $matwin
    }

    method colormapWarning {maptype} {
	toplevel .ui$this.warn
	set yn [tk_dialog .ui$this.warn "Are you sure?" \
		"Current settings will be lost.  \
		Are you sure you want a \"$maptype\" colormap?" \
		warning 0 "Yes" "No"]
	if [expr $yn == 0] {
	    $this-c setmap $maptype
	    $this repaint
	}
    }

    method splineWarning {} {
	if [expr [llength $keys] >= 4] {
	    $this-c interp
	    $this repaint
	    return
	} else {
	    toplevel .ui$this.warn
	    tk_dialog .ui$this.warn "Insufficient Number of Keys" \
		    "You must have at least four keys for \"Spline\" \
		    interpolation." \
		    warning 0 "Ok"
	    global $this-interp_type
	    set $this-interp_type "Linear"
	}
    }
}

proc initColor {c r g b} {
    global $c-r $c-g $c-b
    set $c-r $r
    set $c-g $g
    set $c-b $b
}

proc initMaterial {matter} {
    initColor $matter-ambient 0.0 0.0 0.0
    initColor $matter-diffuse 0.0 0.0 0.0
    initColor $matter-specular 0.0 0.0 0.0
    global $matter-shininess
    set $matter-shininess 10.0
    initColor $matter-emission 0.0 0.0 0.0
    global $matter-reflectivity
    set $matter-reflectivity 0.5
    global $matter-transparency
    set $matter-transparency 0
    global $matter-refraction_index
    set $matter-refraction_index 1.0
}
