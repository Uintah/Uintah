
#
#  ExtractSubmatrix.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   Novemeber 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class ExtractSubmatrix {
    inherit Module
    constructor {config} {
	set name ExtractSubmatrix
	set_defaults
    }
    
    method set_defaults {} {
	global $this-ntrows
	global $this-ntcols
	set $this-ntrows 0
	set $this-ntcols 0
	trace variable $this-ntrows w "$this matrixSizeChanged"
	trace variable $this-ntcols w "$this matrixSizeChanged"
	$this-c needexecute
    }
    
    method raisePlot {} {
#	puts "here!"
	set w .ui$this
	if {[winfo exists $w.submatrix]} {
	    raise $w.submatrix
	} else {
	    toplevel $w.submatrix
	    wm aspect $w.submatrix 1 1 1 1
	    opengl $w.submatrix.gl -geometry 300x300 -doublebuffer false -direct false -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 0
	    bind $w.submatrix.gl <Expose> "$this-c redrawMatrices"
#	    bind $w.submatrix.gl <ButtonPress-1> "$this snapRect %x %y"
#	    bind $w.submatrix.gl <Button1-Motion> "$this snapRect %x %y"
#	    bind $w.submatrix.gl <ButtonRelease-1> "$this 
	    pack $w.submatrix.gl -fill both -expand 1
	}
    }
	
    method snapRect {wx wy} {
#	puts "snapping snoop to $wx $wy"
	global $this-snoopX
	global $this-snoopY
	set $this-snoopX $wx
	set $this-snoopY $wy
	$this-c redrawRect
    }

    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w

	frame $w.f
	frame $w.f.f -relief sunken -bd 1
	frame $w.f.s -relief sunken -bd 1
	label $w.f.f.l -text "Full Matrix"
	global $this-ntrows
	global $this-ntcols
	frame $w.f.f.r
	frame $w.f.f.c
	label $w.f.f.r.l -text "Number of Rows:"
	entry $w.f.f.r.e -width 4 -relief sunken -bd 1 -textvariable $this-ntrows
	pack $w.f.f.r.l $w.f.f.r.e -side left -padx 2 -pady 2
	label $w.f.f.c.l -text "Number of Cols:"
	entry $w.f.f.c.e -width 4 -relief sunken -bd 1 -textvariable $this-ntcols
	pack $w.f.f.c.l $w.f.f.c.e -side left -padx 2 -pady 2
	global $this-always
	set $this-always 1
	checkbutton $w.f.f.d -text "Rebuild on every execute" -variable $this-always
	button $w.f.f.s -text "Explicit Send" -command "$this-c send"	
	button $w.f.f.b -text "Raise/Open Plot" -command "$this raisePlot"
	pack $w.f.f.l $w.f.f.r $w.f.f.c -side top -expand 1
	pack $w.f.f.d $w.f.f.s $w.f.f.b -side top
	label $w.f.s.l -text "Submatrix"
	global $this-minRow
	global $this-maxRow
	global $this-minCol
	global $this-maxCol
	range $w.f.s.r -from 0 -to [set $this-ntrows] -label "Rows: "\
		-showvalue true -var_min $this-minRow -var_max $this-maxRow \
		-orient horizontal -length 250 -command "$this-c redrawRect"
	if {[set $this-ntrows] != 0} {
	    $w.f.s.r config -to [set $this-ntrows]
	    $w.f.s.r config -from 1
	}	
	$w.f.s.r setMinMax 0 [set $this-ntrows]
	range $w.f.s.c -from 0 -to [set $this-ntcols] -label "Columns: "\
		-showvalue true -var_min $this-minCol -var_max $this-maxCol \
		-orient horizontal -length 250 -command "$this-c redrawRect"
	if {[set $this-ntcols] != 0} {
	    $w.f.s.c config -to [set $this-ntcols]
	    $w.f.s.c config -from 1
	}
	$w.f.s.c setMinMax 0 [set $this-ntcols]
	pack $w.f.s.l $w.f.s.r $w.f.s.c -side top -fill both -expand 1
	pack $w.f.f $w.f.s -expand 1 -fill both -side left
	pack $w.f
    }

    method matrixSizeChanged {name element op} {
	global $this-ntrows
	global $this-ntcols
	set w .ui$this
	if {[set $this-ntrows] != 0} {
	    $w.f.s.r config -to [set $this-ntrows]
	    $w.f.s.r config -from 1
	} else {
	    $w.f.s.r config -from 0
	    $w.f.s.r config -to 0
	}
	if {[set $this-ntcols] != 0} {
	    $w.f.s.c config -to [set $this-ntcols]
	    $w.f.s.c config -from 1
	} else {
	    $w.f.s.c config -from 0
	    $w.f.s.c config -to 0
	}
	this-c needexecute
    }
}

