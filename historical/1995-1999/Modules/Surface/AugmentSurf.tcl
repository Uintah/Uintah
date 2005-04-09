
itcl_class AugmentSurf {
    inherit Module
    constructor {config} {
	set name AugmentSurf
	set_defaults
    }
    method set_defaults {} {
        global $this-tclGridSize
	set $this-tclGridSize 16
    }
    method ui {} {
        set w .ui$this
        if {[winfo exists $w]} {
                raise $w
                return;
        }

        toplevel $w
        wm minsize $w 300 20

        global $this-tclGridSize
        scale $w.grid -variable $this-tclGridSize -orient horizontal -from 2 \
		-to 257 -resolution 1 -showvalue true -tickinterval 51 \
	        -digits 0 -label "Grid Size:"
        pack $w.grid -fill x -expand 1 -side top

        button $w.execute -text "Execute" -command "$this-c needexecute"
        pack $w.execute -side top -fill x -pady 2 -padx 2
    }
}
