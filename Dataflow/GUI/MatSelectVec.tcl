##
 #  MatSelectVec.tcl: Select a row or column from a matrix
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   June 1999
 #
 #  Copyright (C) 1999 SCI Group
 # 
 #  Log Information:
 #
 ##

catch {rename MatSelectVec ""}

itcl_class PSECommon_Matrix_MatSelectVec {
    inherit Module
    constructor {config} {
        set name MatSelectVec
        set_defaults
    }
    method set_defaults {} {	
	global $this-colTCL
	global $this-colMaxTCL
	global $this-rowTCL
	global $this-rowMaxTCL
	global $this-rowOrColTCL;
	global $this-animateTCL;
	set $this-colTCL 0
	set $this-colMaxTCL 100
	set $this-rowTCL 0
	set $this-rowMaxTCL 100
	set $this-rowOrColTCL col
        set $this-animateTCL 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 150 30
	set n "$this-c needexecute "

	global $this-colTCL
	global $this-colMaxTCL
	global $this-rowTCL
	global $this-rowMaxTCL
	global $this-rowOrColTCL
	global $this-animateTCL

	frame $w.f
        scale $w.f.r -variable $this-rowTCL \
                -from 0 -to [set $this-rowMaxTCL] \
		-label "Row #" \
                -showvalue true -orient horizontal
        scale $w.f.c -variable $this-colTCL \
                -from 0 -to [set $this-colMaxTCL] \
		-label "Column #" \
                -showvalue true -orient horizontal
	frame $w.f.ff
	radiobutton $w.f.ff.r -text "Row" -variable $this-rowOrColTCL -value row
        radiobutton $w.f.ff.c -text "Column" -variable $this-rowOrColTCL -value col
	pack $w.f.ff.r $w.f.ff.c -side left -fill x -expand 1
	frame $w.f.b
	button $w.f.b.go -text "Execute" -command $n 
	checkbutton $w.f.b.a -text "Animate" -variable $this-animateTCL
        button $w.f.b.stop -text "Stop" -command "$this-c stop" 
	pack $w.f.b.go $w.f.b.a $w.f.b.stop -side left -fill x -expand 1
	pack $w.f.r $w.f.c $w.f.ff $w.f.b -side top -fill x -expand yes
	pack $w.f
    }

    method update {} {
	set w .ui[modname]
        if {[winfo exists $w]} {
	    global $this-colMaxTCL
	    global $this-rowMaxTCL
	    $w.f.r config -to [set $this-colMaxTCL]
	    $w.f.c config -to [set $this-rowMaxTCL]
	    puts "updating!"
	}
    }
}
