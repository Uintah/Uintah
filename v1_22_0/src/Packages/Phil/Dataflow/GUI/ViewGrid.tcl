global CoreTCL
source $CoreTCL/Filebox.tcl

itcl_class Packages/Phil_Tbon_ViewGrid {

    inherit Module
    constructor {config} {
        set name ViewGrid
        set_defaults
    }


    method set_defaults {} {
	set $this-gridtype 0
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

        set n "$this-c needexecute"

	frame $w.top -relief groove -bd 2
	frame $w.mid -relief groove -bd 2
	frame $w.bot -relief groove -bd 2

	frame $w.top.above
	frame $w.mid.below
	frame $w.bot.below

	make_labeled_radio $w.top.above.radio1 "Grid Type" $n top \
		$this-gridtype {{"Regular" 0} {"Curvilinear" 1}}
	make_labeled_radio $w.top.above.radio2 "Representation" $n top \
		$this-representation {{"Points" 0} {"Spheres" 1}}
	scale $w.top.scale1 -label "Sphere Radius" \
		-from 0 -to 1 -resolution 0.001 \
		-variable $this-radius \
		-length 3c -orient horizontal -command $n

	pack $w.top.above.radio1 $w.top.above.radio2 -in $w.top.above \
		-side left -fill both -padx 3
	pack $w.top.above $w.top.scale1 -in $w.top -side top -fill both \
		-padx 2 -pady 2


	label $w.mid.l1 -text "Regular Grid Parameters"

	label $w.mid.below.l1 -text "nx:"
	entry $w.mid.below.e1 -width 5 -relief sunken -bd 2 \
		-textvariable $this-nx1
	label $w.mid.below.l2 -text "ny:"
	entry $w.mid.below.e2 -width 5 -relief sunken -bd 2 \
		-textvariable $this-ny1
	label $w.mid.below.l3 -text "nz:"
	entry $w.mid.below.e3 -width 5 -relief sunken -bd 2 \
		-textvariable $this-nz1

	pack $w.mid.below.l1 $w.mid.below.e1 $w.mid.below.l2 $w.mid.below.e2 \
		$w.mid.below.l3 $w.mid.below.e3 -in $w.mid.below -side left \
		-fill both 
	pack $w.mid.l1 $w.mid.below -in $w.mid -side top -fill both \
		-padx 2 -pady 2

	
	label $w.bot.l1 -text "Curvilinear Grid Parameters"
	button $w.bot.bfile -text "Geom File (CL)" -command "$this fileui"
	scale $w.bot.scale1 -label "Number of Zones:" \
		-from 1 -to 4 -resolution 1 \
		-variable $this-numzones -orient horizontal -length 3c

	set i 1
	while { $i <= 4 } {
	    frame $w.bot.below.f$i
	    checkbutton $w.bot.below.f$i.cb -text "Zone $i" \
		    -variable $this-showzone$i -anchor w
	    label $w.bot.below.f$i.l1 -text "nx:"
	    entry $w.bot.below.f$i.e1 -width 5 -relief sunken -bd 2 \
		    -textvariable $this-nx$i
	    label $w.bot.below.f$i.l2 -text "ny:"
	    entry $w.bot.below.f$i.e2 -width 5 -relief sunken -bd 2 \
		    -textvariable $this-ny$i
	    label $w.bot.below.f$i.l3 -text "nz:"
	    entry $w.bot.below.f$i.e3 -width 5 -relief sunken -bd 2 \
		    -textvariable $this-nz$i

	    pack $w.bot.below.f$i.cb $w.bot.below.f$i.l1 $w.bot.below.f$i.e1 \
		    $w.bot.below.f$i.l2 $w.bot.below.f$i.e2 \
		    $w.bot.below.f$i.l3 $w.bot.below.f$i.e3 \
		    -in $w.bot.below.f$i -side left -fill both
	    pack $w.bot.below.f$i -in $w.bot.below -side top -fill both \
		    -padx 2 -pady 1

	    incr i
	}
	pack $w.bot.l1 $w.bot.bfile $w.bot.scale1 $w.bot.below \
		-in $w.bot -side top \
		-fill both -padx 2 -pady 2

	pack $w.top $w.mid $w.bot -in $w -side top -fill both

    }    
}


