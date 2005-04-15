
itcl_class Uintah_Visualization_ParticleTensorElementExtractor {
    inherit Module

    constructor {config} {
	set name ParticleTensorElementExtractor
	set_defaults
    }

    method set_defaults {} {
	global $this-row
	global $this-column
	global $this-elem
	set $this-row 2
	set $this-column 2
	set $this-elem 5
    }
    method make_entry {w text v c} {
	frame $w
	label $w.l -text "$text"
	pack $w.l -side left -anchor w
	entry $w.e -textvariable $v -width 3
	bind $w.e <Return> $c
	pack $w.e -side right -anchor e
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this select"

	frame $w.m 
	pack $w.m -padx 2 -pady 2 -fill x -expand yes

	frame $w.m.r1
	radiobutton $w.m.r1.c1 -command $n -variable $this-elem -value 1
	radiobutton $w.m.r1.c2 -command $n -variable $this-elem -value 2
	radiobutton $w.m.r1.c3 -command $n -variable $this-elem -value 3
	pack $w.m.r1.c1 $w.m.r1.c2 $w.m.r1.c3 -side left -anchor n
	frame $w.m.r2
	radiobutton $w.m.r2.c1 -command $n -variable $this-elem -value 4
	radiobutton $w.m.r2.c2 -command $n -variable $this-elem -value 5
	radiobutton $w.m.r2.c3 -command $n -variable $this-elem -value 6
	pack $w.m.r2.c1 $w.m.r2.c2 $w.m.r2.c3 -side left -anchor n
	frame $w.m.r3
	radiobutton $w.m.r3.c1 -command $n -variable $this-elem -value 7
	radiobutton $w.m.r3.c2 -command $n -variable $this-elem -value 8
	radiobutton $w.m.r3.c3 -command $n -variable $this-elem -value 9
	pack $w.m.r3.c1 $w.m.r3.c2 $w.m.r3.c3 -side left -anchor n

	pack $w.m.r1 $w.m.r2 $w.m.r3

	make_entry $w.row "Row" $this-row "$this enter"
	make_entry $w.column "Column" $this-column "$this enter"
	pack $w.row $w.column -expand yes -fill x -padx 5

	button $w.b -text Close -command "destroy $w"
	pack $w.b -side bottom -expand yes -fill x -padx 2 -pady 2
    }

    method select {} {
	set w .ui[modname]

	set $this-row [expr ([set $this-elem] + 2) / 3]
	set $this-column [expr ([set $this-elem] - 1) % 3 + 1]

	$this-c needexecute
    }
   
    method enter {} {
	set $this-row [expr ([set $this-row] - 1) % 3 + 1]
	set $this-column [expr ([set $this-column] - 1) % 3 + 1]
	set $this-elem [expr 3*([set $this-row]-1) + [set $this-column]]

	$this-c needexecute
    }
}

