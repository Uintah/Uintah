
itcl_class Uintah_MPMViz_cfdGridLines {
    inherit Module

    method modname {} {
	set n $this
	if {[string first "::" "$n"] == 0} {
	    set n "[string range $n 2 end]"
	}
	return $n
    }

    constructor {config} {
	set name cfdGridLines
	set_defaults
    }

    method set_defaults {} {
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this-c needexecute"
	
	frame $w.m
	make_labeled_radio $w.m.r "Grid Type" $n top \
		$this-mode {{"None" 0} {"Inside" 1 } {"Outside" 2} {"Both" 3}}
	pack $w.m.r

	frame $w.f
	pack $w.m $w.f -padx 2 -pady 2 -fill x -expand yes

	scale $w.f.rad -label "Size of Grid Lines" -orient horizontal \
		-from 0 -to 1 -resolution 0.0001 \
		-length 8c -variable $this-rad -command $n

	frame $w.f.x
	frame $w.f.y
	frame $w.f.x.exp
	frame $w.f.y.exp

	scale $w.f.x.s -label "X Scale Factor" -orient horizontal \
		-from 0 -to 1 -resolution 0.0001 -length 6c \
		-variable $this-scalex -command $n
	label $w.f.x.l -text "E"

	entry $w.f.x.exp.e -width 3 -relief sunken -bd 2 \
		-textvariable $this-expx -state disabled 
	button $w.f.x.exp.bplus -text "+" -command "incr $this-expx; $this-c needexecute"
	button $w.f.x.exp.bminus -text "-" -command "incr $this-expx -1; $this-c needexecute"
	
	pack $w.f.x.exp.bplus $w.f.x.exp.e $w.f.x.exp.bminus \
		-in $w.f.x.exp -side top -ipadx 2
	pack $w.f.x.s $w.f.x.l $w.f.x.exp -in $w.f.x -side left -fill x

	scale $w.f.y.s -label "Y Scale Factor" -orient horizontal \
		-from 0 -to 1 -resolution 0.0001 -length 6c \
		-variable $this-scaley -command $n
	label $w.f.y.l -text "E"

	entry $w.f.y.exp.e -width 3 -relief sunken -bd 2 \
		-textvariable $this-expy -state disabled
	button $w.f.y.exp.bplus -text "+" -command  "incr $this-expy; $this-c needexecute"
	button $w.f.y.exp.bminus -text "-" -command "incr $this-expy -1; $this-c needexecute"
	
	pack $w.f.y.exp.bplus $w.f.y.exp.e $w.f.y.exp.bminus \
		-in $w.f.y.exp -side top -ipadx 2
	pack $w.f.y.s $w.f.y.l $w.f.y.exp -in $w.f.y -side left -fill x

	pack $w.f.rad $w.f.x $w.f.y -side top -fill x
    }
}

