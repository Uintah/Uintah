
#
#  VisualizeMatrix.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   Novemeber 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class VisualizeMatrix {
    inherit Module
    constructor {config} {
	set name VisualizeMatrix
	set_defaults
    }

#    trace variable $this-array w "$this matChanged"
    
    method set_defaults {} {
	$this-c needexecute
	global $this-numMatrices
	global $this-snoopOn
	global $this-snoopRender
	set $this-numMatrices 0
	set $this-snoopOn 0
	set $this-snoopRender uniform_dots
    }
    
    method typeChanged {name element op} {
	$this-c send
    }

    # this method is only useful is the plot window is a toplevel window
    # if it's not, no harm will be done -- this just won't ever get called
    method raisePlot {} {
#	puts "here!"
	set w .ui[modname]
	if {[winfo exists $w.plot]} {
	    raise $w.plot
	} else {
	    toplevel $w.plot
	    wm aspect $w.plot 1 1 1 1
	    puts "just before opengl #3"
	    opengl $w.plot.gl -geometry 300x300 -doublebuffer true -direct true\
		 -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2 -visual 2
	    bind $w.plot.gl <Expose> "$this-c redrawMatrices"
	    bind $w.plot.gl <ButtonPress-1> "$this snapSnoop %x %y"
	    bind $w.plot.gl <Button1-Motion> "$this snapSnoop %x %y"
	    bind $w.plot.gl <ButtonPress-2> "$this toggleSnoop"	
	    pack $w.plot.gl -fill both -expand 1
	}
	if {[winfo exists $w.plot.snoop]} {
	    raise $w.plot.snoop
	} else {
	    toplevel $w.plot.snoop
	    wm aspect $w.plot.snoop 1 1 1 1
	    puts "just before opengl #4"
	    opengl $w.plot.snoop.gl -geometry 300x300 -doublebuffer true -direct true\
		 -rgba true -redsize 2 -greensize 2 -bluesize 2 -depthsize 2 -visual 2
	    bind $w.plot.snoop.gl <Expose> "$this-c redrawSnoop"
	    bind $w.plot.snoop.gl <Destroy> "$this-c snoop_dying"
	    pack $w.plot.snoop.gl -fill both -expand 1
	}
    }
	
    method snapSnoop {wx wy} {
#	puts "snapping snoop to $wx $wy"
	global $this-snoopX
	global $this-snoopY
	set $this-snoopX $wx
	set $this-snoopY $wy
	$this-c redrawSnoop
    }

    method toggleSnoop {} {
#	puts "toggling snoop"
	global $this-snoopOn
	if {[set $this-snoopOn] == 1} {
	    set $this-snoopOn 0
	} else {
	    set $this-snoopOn 1
	}
	$this-c redrawSnoopRect
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w		    
	    return;
	}
	toplevel $w

	frame $w.f
	global $this-numMatrices
	trace variable $this-numMatrices w "$this numMatricesChanged"
	frame $w.f.gen
	frame $w.f.gen.sn -relief sunken -bd 1
	frame $w.f.gen.pl -relief sunken -bd 1
	label $w.f.gen.sn.l -text "Snoop Configuration"
	global $this-snoopRender
        make_labeled_radio $w.f.gen.sn.r "Render Options:" \
		"$this-c redrawSnoop"\
                top "$this-snoopRender"\
                {{"Scaled Dots" scaled_dots} \
                {"Uniform Dots" uniform_dots} \
		{"Scaled Text" scaled_text} \
	        {"Uniform Text" uniform_text}}
	pack $w.f.gen.sn.l $w.f.gen.sn.r -side top
	pack $w.f.gen.sn -side top -fill both -expand 1
	label $w.f.gen.pl.l -text "Global Operations"
	button $w.f.gen.pl.print -text "Print Values" -command "$this print"
	button $w.f.gen.pl.raiseplot -text "Raise/Open Plot" -command "$this raisePlot"
	pack $w.f.gen.pl.l $w.f.gen.pl.print $w.f.gen.pl.raiseplot -side top
	pack $w.f.gen.pl -side bottom -fill both
	pack $w.f.gen -side left -fill both
	for { set i 0 } { $i < [set $this-numMatrices] } { incr i } {
	    frame $w.f.m$i -relief sunken -bd 1
	    label $w.f.m$i.l -text "-- Matrix $i --"
	    frame $w.f.m$i.row
	    label $w.f.m$i.row.l -text "Rows:"
	    global $this-nrow$i
	    entry $w.f.m$i.row.e -width 8 -relief sunken -bd 1 -textvariable $this-nrow$i
	    pack $w.f.m$i.row.l $w.f.m$i.row.e -side left -padx 2 -pady 2
	    frame $w.f.m$i.col
	    label $w.f.m$i.col.l -text "Columns:"
	    global $this-ncol$i
	    entry $w.f.m$i.col.e -width 8 -relief sunken -bd 1 -textvariable $this-ncol$i
	    pack $w.f.m$i.col.l $w.f.m$i.col.e -side left -padx 2 -pady 2
	    frame $w.f.m$i.type
	    label $w.f.m$i.type.l -text "Type:"
	    global $this-type$i
	    entry $w.f.m$i.type.e -width 20 -relief sunken -bd 1 -textvariable $this-type$i
	    pack $w.f.m$i.type.l $w.f.m$i.type.e -side left -padx 2 -pady 2
	    frame $w.f.m$i.density
	    label $w.f.m$i.density.l -text "1./Density:"
	    global $this-density$i
	    entry $w.f.m$i.density.e -width 10 -relief sunken -bd 1 -textvariable $this-density$i
	    pack $w.f.m$i.density.l $w.f.m$i.density.e -side left -padx 2 -pady 2
	    frame $w.f.m$i.condition
	    label $w.f.m$i.condition.l -text "Condition:"
	    global $this-condition$i
	    entry $w.f.m$i.condition.e -width 10 -relief sunken -bd 1 -textvariable $this-condition$i
	    pack $w.f.m$i.condition.l $w.f.m$i.condition.e -side left -padx 2 -pady 2
	    global $this-symm$i
	    global $this-posdef$i
	    set $this-symm$i "Symmetric"
	    set $this-posdef$i "Positive Definite"
	    label $w.f.m$i.symm -textvariable $this-symm$i
	    label $w.f.m$i.posdef -textvariable $this-posdef$i
	    global $this-scale$i
	    checkbutton $w.f.m$i.scale -text "Scale to fit?" -variable $this-scale$i -command "$this-c needexecute" -anchor w
	    global $this-shown$i
	    checkbutton $w.f.m$i.shown -text "Plot it?" -variable $this-shown$i -command "$this-c needexecute" -anchor w
	    pack $w.f.m$i.l $w.f.m$i.row $w.f.m$i.col $w.f.m$i.type $w.f.m$i.density $w.f.m$i.condition $w.f.m$i.symm $w.f.m$i.posdef $w.f.m$i.scale $w.f.m$i.shown -side top -expand 1

	    pack $w.f.m$i -side left
	}
	pack $w.f -side bottom
	
	if {[winfo exists $w.plot]} {
	    raise $w.plot
	} else {
	    # now build the plot window

	    # START: uncomment these two if plot is *not* a separate window
	    #frame $w.plot -relief raised -bd 1
	    #pack $w.plot -side top -expand 1 -fill both
	    # END:
	    
	    # START: these next two lines build a topleve plot win w/ aspect=1
	    toplevel $w.plot
	    wm aspect $w.plot 1 1 1 1
	    # END
	    puts "just before opengl #1"
	    opengl $w.plot.gl -geometry 300x300 -doublebuffer true -direct true\
		 -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2 -visual 2
	    bind $w.plot.gl <Expose> "$this-c redrawMatrices"
	    bind $w.plot.gl <ButtonPress-1> "$this snapSnoop %x %y"
	    bind $w.plot.gl <Button1-Motion> "$this snapSnoop %x %y"
	    bind $w.plot.gl <ButtonPress-2> "$this toggleSnoop"	
	    pack $w.plot.gl -fill both -expand 1
	}
	if {[winfo exists $w.plot.snoop]} {
	    raise $w.plot.snoop
	} else {
	    toplevel $w.plot.snoop
	    wm aspect $w.plot.snoop 1 1 1 1
	    puts "just before opengl #2"
	    opengl $w.plot.snoop.gl -geometry 300x300 -doublebuffer true -direct true\
		 -rgba true -redsize 1 -greensize 1 -bluesize 1 -depthsize 2 -visual 2
	    bind $w.plot.snoop.gl <Expose> "$this-c redrawSnoop"
	    bind $w.plot.snoop.gl <Destroy> "$this-c snoop_dying"
	    pack $w.plot.snoop.gl -fill both -expand 1
	}
    }

    method print { } {
	global $this-numMatrices
	puts -nonewline "Number of matrices: "
	puts [set $this-numMatrices]
	for { set i 0 } { $i < [set $this-numMatrices] } { incr i } {
	    puts "Matrix $i: "
	    global $this-nrow$i
	    puts -nonewline "   Rows: "
	    puts [set $this-nrow$i]
	    global $this-ncol$i
	    puts -nonewline "   Columns: "
	    puts [set $this-ncol$i]
	    global $this-type$i
	    puts -nonewline "   Type: "
	    puts [set $this-type$i]
	    global $this-density$i
	    puts -nonewline "   Density: "
	    puts [set $this-density$i]
	    global $this-condition$i
	    puts -nonewline "   Condition: "
	    puts [set $this-condition$i]
	    global $this-symm$i
	    puts -nonewline "   Symmetric: "
	    puts [set $this-symm$i]
	    global $this-posdef$i
	    puts -nonewline "   Positive Definite: "
	    puts [set $this-posdef$i]
	    global $this-scale$i
	    puts -nonewline "   Scale: "
	    puts [set $this-scale$i]
	    global $this-shown$i
	    puts -nonewline "   Shown: "
	    puts [set $this-shown$i]
	}
    }
    
    method numMatricesChanged {name element op} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    destroy $w
#	    ui
	    return;
	}
    }
}

