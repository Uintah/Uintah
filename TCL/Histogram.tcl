#
#  Histogram.tcl
#
#  Written by:
#   James T. Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class Histogram {
    public title "Untitled"
    public valtitle "Values"
    public freqtitle "Freq."
    public minval 0
    public maxval 0
    # `freqs' is a list of integers
    public freqs {}
    public grid no
    public range no

    # If min/maxfreq are NOT provided, set calcminmax to yes.
    public calcminmax no
    public minfreq 0
    public maxfreq 0

    constructor {config} {
	set canvasx 745
	set canvasy 410
	set xmin 60
	set ymin 60
	set xmax [expr $canvasx-45]
	set ymax [expr $canvasy-45]
	set yrange [expr $ymax-$ymin]
	set xrange [expr $xmax-$xmin]

	global $this-rangeleft $this-rangeright
	set $this-rangeleft 0
	set $this-rangeright 0
    }

    method config {config} {
    }

    protected canvasx
    protected canvasy
    protected xmin
    protected ymin
    protected xmax
    protected ymax
    protected xrange
    protected yrange

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
	set hist $w.f
	
	canvas $hist.canvas -scrollregion {0 0 745 410} \
		-width 745 -height 410
	bind $hist.canvas <Motion> "$this mouse %x"
	pack $hist.canvas -side left -padx 2 -pady 2 -fill both -expand yes
	pack $hist -fill both -expand yes

	$this update
    }

    method update {} {
	$this MinMax
	set xdelt [expr $xrange/double($datasize-1)]
	set xnum 30
	set ynum 20

	.ui$this.f.canvas delete all

	.ui$this.f.canvas create text [expr $canvasx/2.0] 0 \
		-text $title -anchor n -justify center \
		-font "-Adobe-Helvetica-bold-R-Normal--*-180-100-*"
	.ui$this.f.canvas create text [expr $canvasx/2.0] [expr $canvasy-4] \
		-text $valtitle -anchor s -justify center
	.ui$this.f.canvas create text 5 [expr $ymin-10] \
		-text $freqtitle -anchor sw -justify left

	if [expr (([string compare $grid "y"]==0) \
		|| ([string compare $grid "yes"]==0))] {
	    for {set i 0} {$i <= $ynum} {incr i 1} {
		set y [expr $ymax-$i*$yrange/double($ynum)]
		.ui$this.f.canvas create line $xmin $y $xmax $y
	    }
	    for {set i 0} {$i <= $xnum} {incr i 1} {
		set x [expr $xmin+$i*($xrange-$xdelt)/double($xnum)]
		.ui$this.f.canvas create line $x $ymin $x $ymax
	    }
	}
	
	if [expr $datasize == 0] {
	    return
	}

	set x $xmin
	for {set i 0} {$i < $datasize} {incr i 1} {
	    set val [expr $minval+$i*$valrange/($datasize-1)]
	    set freq [lindex $freqs $i]
	    set oldx $x
	    set x [expr $xmin+($val-$minval)*$xrange/double($valrange)]
	    set y [expr $ymax-$freq*$yrange/double($maxfreq)]
	    .ui$this.f.canvas create rectangle \
		    $oldx $ymax $x $y -tags rects\
		    -fill #224488 \
		    -outline #224488
	    if [expr $datasize < 20] {
		.ui$this.f.canvas create text [expr $oldx+$xdelt/2.0] $y \
			-anchor s -justify center -text $freq
	    }
	}
	set xnum [$this Min 6 $datasize]
	for {set i 0} {$i <= $xnum} {incr i 1} {
	    .ui$this.f.canvas create text [expr $xmin+$i*$xrange/double($xnum)] \
		    [expr $ymax+4] -anchor n -justify center \
		    -text [format %3.2f [expr $minval+$i*$valrange/double($xnum)]]
	}
	for {set i 0} {$i <= $ynum} {incr i 1} {
	    set y [expr $ymax-$i*$yrange/double($ynum)]
	    .ui$this.f.canvas create text [expr $xmin-4] $y \
		    -anchor e -justify right \
		    -text [format %3.2f [expr $minfreq+$i*$freqrange/double($ynum)]]
	}
	.ui$this.f.canvas create text [expr $canvasx/2.0] $ymin -tags text \
		-text "" -anchor s -justify center

	if [expr (([string compare $range "y"]==0) \
		|| ([string compare $range "yes"]==0))] {
	    set i1 [expr int(($rangeleft-$xmin)*($datasize-1)/double($xrange))]
	    set i2 [expr int(($rangeright-$xmin)*($datasize-1)/double($xrange))]
	    set val1 [expr $minval+$i1*$valrange/($datasize-1)]
	    set val2 [expr $minval+$i2*$valrange/($datasize-1)]
	    .ui$this.f.canvas create text 4 [expr $canvasy-4] -tags range \
		    -text "Range($val1--$val2)" -anchor sw -justify left
	    
	    .ui$this.f.canvas create rectangle \
		    [expr $rangeleft-5] $ymax $rangeleft $ymin \
		    -tags left \
		    -fill #ffff00000000 \
		    -outline #ffff00000000
	    .ui$this.f.canvas create rectangle \
		    $rangeright $ymax [expr $rangeright+5] $ymin \
		    -tags right \
		    -fill #ffff00000000 \
		    -outline #ffff00000000
	    .ui$this.f.canvas bind left <Button1-Motion> \
		    "$this mouseleft %x"
	    .ui$this.f.canvas bind right <Button1-Motion> \
		    "$this mouseright %x"

	    $this repaint
	}
    }

    protected rangeleft 200
    protected rangeright 300

    method leftval {} {
	set i1 [expr int(($rangeleft-$xmin)*($datasize-1)/double($xrange))]
	return [expr $minval+$i1*$valrange/($datasize-1)]
    }

    method rightval {} {
	set i2 [expr int(($rangeright-$xmin)*($datasize-1)/double($xrange))]
	return [expr $minval+$i2*$valrange/($datasize-1)]
    }

    method mouse {x} {
	if [expr (($x >= $xmin) && ($x <= $xmax))] {
	    set i [expr int(($x-$xmin)*($datasize-1)/double($xrange))]
	    set val [expr $minval+$i*$valrange/($datasize-1)]
	    set freq [lindex $freqs $i]
	    .ui$this.f.canvas itemconfigure text \
		    -text "$valtitle\($val) $freqtitle\($freq)"
	} else {
	    .ui$this.f.canvas itemconfigure text -text ""
	}
    }

    method mouseleft {x} {
	if [expr (($x >= $xmin) && ($x <= $xmax))] {
	    if [expr $x >= $rangeright] {
		set x $rangeright
	    }
	    set rangeleft $x
	    .ui$this.f.canvas coords left \
		    [expr $rangeleft-5] $ymax $rangeleft $ymin
	    $this repaint
	}
    }

    method mouseright {x} {
	if [expr (($x >= $xmin) && ($x <= $xmax))] {
	    if [expr $x <= $rangeleft] {
		set x $rangeleft
	    }
	    set rangeright $x
	    .ui$this.f.canvas coords right \
		    $rangeright $ymax [expr $rangeright+5] $ymin
	    $this repaint
	}
    }

    method repaint {} {
	set i1 [expr int(($rangeleft-$xmin)*($datasize-1)/double($xrange))]
	set i2 [expr int(($rangeright-$xmin)*($datasize-1)/double($xrange))]
	set val1 [expr $minval+$i1*$valrange/($datasize-1)]
	set val2 [expr $minval+$i2*$valrange/($datasize-1)]
	.ui$this.f.canvas itemconfigure range \
		-text "Range($val1--$val2)"

	global $this-rangeleft $this-rangeright
	set $this-rangeleft $val1
	set $this-rangeright $val2

	set lst1 [.ui$this.f.canvas find withtag rects]
	# Use a flat overlap rectangle to get almost only rectangles
	set lst2 [.ui$this.f.canvas find overlapping \
		$rangeleft [expr $ymax-1] $rangeright [expr $ymax-2]]
	foreach rect $lst1 {
	    if [expr [lsearch $lst2 $rect] > -1] {
		.ui$this.f.canvas itemconfigure $rect \
			-fill #000000000000 \
			-outline #000000000000
	    } else {
		.ui$this.f.canvas itemconfigure $rect \
			-fill #224488 \
			-outline #224488
	    }
	}
    }

    protected datasize 0
    protected valrange 0
    protected freqrange 0

    method MinMax {} {
	set datasize [llength $freqs]
	if [expr $datasize == 0] {
	    return
	}

	if [expr (([string compare $calcminmax "y"]==0) \
		|| ([string compare $calcminmax "yes"]==0))] {
	    set minfreq [lindex $freqs 0]
	    set maxfreq $minfreq
	    
	    foreach freq $freqs {
		if [expr $freq < $minfreq] {
		    set minfreq $freq
		} elseif [expr $freq > $maxfreq] {
		    set maxfreq $freq
		}
	    }
	}

	set valrange [expr $maxval-$minval]
	set freqrange [expr $maxfreq-$minfreq]
    }

    method Min {x1 x2} {
	if [expr $x1 < $x2] {
	    return $x1
	} else {
	    return $x2
	}
    }
}