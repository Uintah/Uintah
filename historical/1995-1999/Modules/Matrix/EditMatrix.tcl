
#
#  EditMatrix.tcl
#
#  Written by:
#   David Weinstein
#   Department of Computer Science
#   University of Utah
#   Novemeber 1995
#
#  Copyright (C) 1995 SCI Group
#

itcl_class EditMatrix {
    inherit Module
    constructor {config} {
	set name EditMatrix
	set_defaults
    }

    method matrix_initialize { } {
	global $this-nrow
	global $this-ncol
	global $this-array
	global $this-TclMat

	if { [info exists $this-TclMat] == 1 } { unset $this-TclMat }
	for { set r 0 } { $r < [set $this-nrow] } { incr r } {
	    for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		if { $r == $c } { set $this-array($r,$c) 1.0; lappend $this-TclMat 1.0 } { set $this-array($r,$c) 0.0; lappend $this-TclMat 0.0 }
	    } 
	} 
	for { set r 0 } { $r < [set $this-nrow] } {incr r } {
	    set $this-array($r,-1) -[set r]-
	}
	for { set c 0 } { $c < [set $this-ncol] } {incr c } {
	    set $this-array(-1,$c) -[set c]-
	}
	trace variable $this-array w "$this matChanged"
    }

    method set_defaults {} {
	global $this-Matrixtype
	set $this-Matrixtype dense
	global $this-nrow
	set $this-nrow 5
	global $this-ncol
	set $this-ncol 5
	global $this-array
	global $this-loading
	set $this-loading 0
	matrix_initialize
	$this-c needexecute
    }
    
    method typeChanged {name element op} {
	$this-c send
    }

    method matChanged {name element op} {
	global $this-TclMat
	global $this-ncol
	global $this-array
#	puts -nonewline "$this-nrow is "
#	puts [set $this-nrow]
#	puts " ${name}($element) changed!"
	set comma [string first , $element]
#	puts "comma is $comma"
	set r [string range $element 0 [expr $comma - 1]]
#	puts "r is $r"
	set c [string range $element [expr $comma + 1] end]
#	puts "c is $c"
	set idx [expr [set $this-ncol] * $r + $c]
#	puts "Calculated the changed index to be: $idx"
	set $this-TclMat [lreplace [set $this-TclMat] $idx $idx [set $this-array($element)]]
#	puts -nonewline "$this-TclMat is"
#	puts [set $this-TclMat]
    }

    method dimsChanged { } {
	global $this-nrow
	global $this-nnrow
	global $this-ncol
	global $this-nncol
	set r [set $this-nrow]
	set rr [set $this-nnrow]
	set c [set $this-ncol]
	set cc [set $this-nncol]
#	puts "r: $r  c: $c  rr: $rr  cc: $cc"
	if { $r != $rr } { set $this-nrow $rr; nrChanged }
	if { $c != $cc } { set $this-ncol $cc; ncChanged }
    }
    
    method ncChanged { } {
	global $this-nrow
	global $this-ncol
	global $this-array
	global $this-TclMat
	global $this-loading
	global .ui$this.matrix.tab
puts "changing the columns..."
	set last_nc [expr [lindex [.ui$this.matrix.tab config -cols] 4] - 1]
	trace vdelete $this-array w "$this matChanged"
	.ui$this.matrix.tab config -cols [expr [set $this-ncol] + 1]
	# gotta go in and change $this-array to fit the new dimensions
	set oldMat [set $this-TclMat]
	if { $last_nc < [set $this-ncol] } {
	    # we're growing, fill array with zeros
	    for { set c $last_nc } { $c < [set $this-ncol] } { incr c } {
		for { set r 0 } { $r < [set $this-nrow] } { incr r } {
		    if { [set $this-loading] != 1 } {
			set $this-array($r,$c) 0.0
		    }
		} 
		set $this-array(-1,$c) -[set c]-
	    } 
	    unset $this-TclMat
	    for { set r 0 } { $r < [set $this-nrow] } { incr r } {
		for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		    lappend $this-TclMat [set $this-array($r,$c)]
		}
	    }
	} {
	    # we're shrinking -- just need to update $this-TclMat
	    unset $this-TclMat
	    for { set r 0 } { $r < [set $this-nrow] } { incr r } {
		for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		    lappend $this-TclMat [set $this-array($r,$c)]
		}
	    }
	}
	trace variable $this-array w "$this matChanged"
    }

    method nrChanged { } {
	global $this-nrow
	global $this-ncol
	global $this-array
	global $this-TclMat
	global $this-loading
	global .ui$this.matrix.tab
puts "changing the rows..."
	set last_nr [expr [lindex [.ui$this.matrix.tab config -rows] 4] - 1]
	trace vdelete $this-array w "$this matChanged"
	.ui$this.matrix.tab config -rows [expr [set $this-nrow] + 1]
	# gotta go in and change $this-array to fit the new dimensions
	set oldMat [set $this-TclMat]
	if { $last_nr < [set $this-nrow] } {
	    # we're growing, fill array with zeros
	    for { set r $last_nr } { $r < [set $this-nrow] } { incr r } {
		for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		    if { [set $this-loading] != 1 } {
			set $this-array($r,$c) 0.0
		    }
		} 
		set $this-array($r,-1) -[set r]-
	    } 
	    unset $this-TclMat
	    for { set r 0 } { $r < [set $this-nrow] } { incr r } {
		for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		    lappend $this-TclMat [set $this-array($r,$c)]
		}
	    }
	} {
	    # we're shrinking -- just need to update $this-TclMat
	    unset $this-TclMat
	    for { set r 0 } { $r < [set $this-nrow] } { incr r } {
		for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		    lappend $this-TclMat [set $this-array($r,$c)]
		}
	    }
	}
	trace variable $this-array w "$this matChanged"
    }

    method print { } {
	global $this-nrow
	global $this-ncol
	global $this-array
	global $this-TclMat
	puts "Here's the array:"
	for { set r 0 } { $r < [set $this-nrow] } { incr r } {
	    puts -nonewline "\t"
	    for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		puts -nonewline [format "%6.2f" [set $this-array($r,$c)]]
		puts -nonewline "  "
	    } 
	    puts ""
	} 
	puts "Here's the TclMat:"
	set cnt 0
	for { set r 0 } { $r < [set $this-nrow] } { incr r } {
	    puts -nonewline "\t"
	    for { set c 0 } { $c < [set $this-ncol] } { incr c } {
		puts -nonewline [format "%6.2f" [lindex [set $this-TclMat] $cnt]]
		puts -nonewline "  "
		incr cnt
	    } 
	    puts ""
	} 

    }

#    method tcl_send { } {
#	global $this-TclMat
#	global $this-nrow
#	global $this-ncol
#	global $this-array
#	if { [info exists $this-TclMat] != 0 } { unset $this-TclMat }
#	for { set r 0 } { $r < [set $this-nrow] } { incr r } {
#	    for { set c 0 } { $c < [set $this-ncol] } { incr c } {
#		lappend $this-TclMat [set $this-array($r,$c)]
#	    } 
#	} 
#	$this-c send
#    }

    method tcl_load { } {
	global $this-TclMat
	global $this-nrow
	global $this-ncol
	global $this-array
	global $this-nnrow
	global $this-nncol
#	puts "In tcl_load"
	# rt, ct  get old size	
	set rt [set $this-nnrow]
	set ct [set $this-nncol]
	# the ui gets the new size
	set $this-nnrow [set $this-nrow] 
	set $this-nncol [set $this-ncol]
	# rewrite these to be the old size
	set $this-nrow $rt
	set $this-ncol $ct
	if { [info exists $this-TclMat] == 0 } { return }
	trace vdelete $this-array w "$this matChanged"
	.ui$this.matrix.tab flash mode off
	set cc [set $this-nncol]
	set r 0
	set c 0
	foreach val [set $this-TclMat] {
	    if { $c >= $cc } { incr r; set c 0 }
	    set $this-array($r,$c) $val
	    incr c
	}
#	set mm [set $this-TclMat]
#	set cc [set $this-nncol]
#	set ind 0
#	for { set r 0 } { $r < [set $this-nnrow] } { incr r } {
#	    for { set c 0 } { $c < $cc } { incr c; incr ind } {
#		set $this-array($r,$c) [lindex $mm $ind]
#	    }   
#	}          
	trace variable $this-array w "$this matChanged"
	global .ui$this.matrix.tab
	.ui$this.matrix.tab config -rows [expr [set $this-nrow] + 1]
	.ui$this.matrix.tab config -cols [expr [set $this-ncol] + 1]
	global $this-loading
	set $this-loading 1
puts "done setting up array, now to fix the table..."
	dimsChanged
puts "done with tcl_load, just gotta update..."
	set $this-loading 0
	.ui$this.matrix.tab flash mode on
        update
puts "made it out alive!"
    }

    method ui {} {
	set w .ui$this
		if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	toplevel $w.matrix

	global $this-nrow
	global $this-ncol
	global $this-array
	table $w.matrix.tab -rows [expr [set $this-nrow] + 1] \
		-cols [expr [set $this-ncol] + 1] \
		-ysc [list $w.matrix.sv set] -xscr [list $w.matrix.sh set] \
		-var $this-array -relief flat \
		-roworigin -1 -colorigin -1 \
		-maxwidth 300 -maxheight 300
	scrollbar $w.matrix.sv -comm [list $w.matrix.tab toprow]
	scrollbar $w.matrix.sh -com  [list $w.matrix.tab leftcol] \
		-orient horizontal
	pack $w.matrix.sh -side bottom -fill x
	pack $w.matrix.sv -side right -fill y
	update
	pack $w.matrix.tab -side left -fill both -expand on
	update
	$w.matrix.tab configure -rowt 1 
	update
	$w.matrix.tab configure -colt 1 
	update
	$w.matrix.tab configure -relief raised
	$w.matrix.tab tag con Flash -bg red -fg white
	$w.matrix.tab flash tag Flash
	$w.matrix.tab flash mode on
	$w.matrix.tab draw fast
	update

	wm minsize $w 300 20
	set n "$this-c needexecute "

	global $this-Matrixtype
	global $this-nnrow
	set $this-nnrow [set $this-nrow]
	global $this-nncol
	set $this-nncol [set $this-ncol]
	frame $w.n -relief sunken -bd 1
	frame $w.s -relief sunken -bd 1
	pack $w.n -side top -fill both
	pack $w.s -side top -fill x
	frame $w.n.w -relief raised -bd 1
	frame $w.n.e -relief raised -bd 1
	pack $w.n.w $w.n.e -side left -fill both -expand 1
	make_labeled_radio $w.n.w.type "Matrix Type:" $n \
		top $this-Matrixtype \
		{{Dense dense} {SymSparse symsparse}}
	label $w.n.e.l -text "Matrix Dimensions:"
	frame $w.n.e.r
	label $w.n.e.r.l -text "Number of Rows:"
	entry $w.n.e.r.e -width 4 -relief sunken -bd 1 -textvariable $this-nnrow
	pack $w.n.e.r.l $w.n.e.r.e -side left -padx 2 -pady 2
	frame $w.n.e.c
	label $w.n.e.c.l -text "Number of Cols:"
	entry $w.n.e.c.e -width 4 -relief sunken -bd 1 -textvariable $this-nncol
	pack $w.n.e.c.l $w.n.e.c.e -side left -padx 2 -pady 2
	pack $w.n.e.l $w.n.e.r $w.n.e.c -side top -expand 1 -fill both
	trace variable $this-MatrixType w "$this typeChanged"
	pack $w.n.w.type -side top -padx 5 -anchor w
	button $w.s.p -text "Print Matrix" -command "$this print"
	button $w.s.s -text "Load Matrix" -command "$this-c load"
	button $w.s.l -text "Send Matrix" -command "$this-c send"
	button $w.s.u -text "Update Dims" -command "$this dimsChanged"
	pack $w.s.p $w.s.s $w.s.l $w.s.u -side left -expand 1 
    }
}

