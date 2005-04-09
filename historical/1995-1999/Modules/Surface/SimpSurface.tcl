itcl_class SimpSurface {
    inherit Module
    constructor {config} {
        set name SimpSurface
        set_defaults
    }

    method set_defaults {} {
	global $this-numfaces
	global $this-collapsemode
	set $this-numfaces 10000
	set $this-collapsemode 3
    }

    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2
	make_labeled_radio $w.type "Node Optimization" ""\
                top $this-collapsemode \
                {{"Either End" 0}  \
		{"Ends Or Middle" 1} \
		{"Optimized Along Edge" 2} \
		{"Fully Optimized" 3}}
	scale $w.f.s -variable $this-numfaces -orient horizontal \
                -from 100 -to 50000 -showvalue true

	button $w.f.e -text "Execute" -command "$this-c needexecute"

	pack $w.f.s $w.f.e $w.type -fill x
    }
}
