itcl_class Tbon {

    inherit Module
    constructor {config} {
        set name Tbon
        set_defaults
    }


    method set_defaults {} {
	set $this-miniso 0
	set $this-maxiso 1
	set $this-res 0.001
	set $this-timesteps 10
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }


        toplevel $w
        wm minsize $w 100 50

        set n "$this-c needexecute"

	scale $w.s1 -label "MC Case" \
		-from 0 -to 255 -variable $this-timevalue \
		-length 10c -orient horizontal -command $n

	pack $w.s1 -in $w -side top -fill both

    }

    
}


