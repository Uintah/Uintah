global CoreTCL
source $CoreTCL/Filebox.tcl

itcl_class Packages/Phil_Tbon_ViewMesh {

    inherit Module
    constructor {config} {
        set name ViewMesh
        set_defaults
    }


    method set_defaults {} {
	set $this-representation 0
	set $this-radius 0.0
    }

    method fileui {} {
	set meta .ui1[modname]
	if {[winfo exists $meta]} {
	    raise $meta
	    return;
	}

	toplevel $meta
	makeFilebox $meta $this-geomfilename \
		"$this-c needexecute" "destroy $meta"
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }


        toplevel $w
        wm minsize $w 100 50

        set n "$this-c update"

	
	button $w.bfile -text ".htvol File" -command "$this fileui"
	make_labeled_radio $w.radio1 "Representation" $n top \
		$this-representation {{"Lines" 0} {"Cylinders" 1}}
	scale $w.scale1 -label "Cylinder Radius" \
		-from 0 -to 1 -resolution 0.001 \
		-variable $this-radius \
		-length 3c -orient horizontal -command $n

	pack $w.bfile $w.radio1 $w.scale1 -in $w -side top \
		-ipadx 2 -ipady 2 -padx 2 -pady 2
    }    
}


