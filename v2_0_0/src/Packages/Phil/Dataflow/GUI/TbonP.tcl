global CoreTCL
source $CoreTCL/Filebox.tcl

itcl_class Packages/Phil_Tbon_TbonP {

    inherit Module
    constructor {config} {
        set name TbonP
        set_defaults
    }


    method set_defaults {} {
	set $this-miniso 0
	set $this-maxiso 1
	set $this-res 0.001
	set $this-timesteps 10
    }

    method metaui {} {
	set meta .ui1[modname]
	if {[winfo exists $meta]} {
	    raise $meta
	    return;
	}

	toplevel $meta
	makeFilebox $meta $this-metafilename \
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
	set cc "$this-c changeColor"

	frame $w.color -relief groove -bd 2
	frame $w.top -relief groove -bd 2
	frame $w.mid -relief groove -bd 2
	frame $w.bot -relief groove -bd 2

	label $w.title -text "Parallel Tbon Parameters"
	button $w.bmeta -text "Metafile..." -command "$this metaui"

	frame $w.color.top
	frame $w.color.bot

	label $w.color.top.lr -text "Red:"
	entry $w.color.top.r -textvariable $this-red -width 5
	label $w.color.top.lg -text "Green:"
	entry $w.color.top.g -textvariable $this-green -width 5
	label $w.color.top.lb -text "Blue:"
	entry $w.color.top.b -textvariable $this-blue -width 5

	label $w.color.bot.la -text "Alpha:"
	entry $w.color.bot.a -textvariable $this-alpha -width 5
	label $w.color.bot.filler -text "        "
	button $w.color.bot.changer -text "Change Color" -command $cc

	pack $w.color.top.lr $w.color.top.r $w.color.top.lg $w.color.top.g \
		$w.color.top.lb $w.color.top.b \
		-in $w.color.top -side left -padx 2 -fill x
	pack $w.color.bot.la $w.color.bot.a \
		$w.color.bot.filler $w.color.bot.changer \
		-in $w.color.bot -side left -padx 2 -fill x
	pack $w.color.top $w.color.bot -in $w.color -side top -pady 2

	make_labeled_radio $w.top.radio1 "Brick Size (Nodes):" {} top \
		$this-nodebricksize {{128 128} {256 256} {512 512} {1024 1024} {2048 2048} {4096 4096}}
	make_labeled_radio $w.top.radio2 "Brick Size (Data):" {} top \
		$this-databricksize {{"256 (4x4x4)" 256} {"2048 (8x8x8)" 2048} {"16384 (16x16x16)" 16384}}

	scale $w.mid.s1 -label "Isovalue" \
		-from 0 -to 1 -resolution 0.01 \
		-variable $this-isovalue \
		-length 3c -orient horizontal -command $n


	scale $w.bot.s1 -label "Time Step" \
		-from 0 -to 10  \
		-variable $this-timevalue \
		-length 3c -orient horizontal -command $n

	scale $w.bot.s2 -label "Resolution" \
		-from 0 -to 10 \
		-variable $this-resolution \
		-length 3c -orient horizontal -command $n

	pack $w.top.radio1 $w.top.radio2 -in $w.top -side left -fill both
	pack $w.mid.s1 -in $w.mid \
		-side top -fill x
	pack $w.bot.s1 $w.bot.s2 -in $w.bot \
		-side top -fill both

	pack $w.title $w.bmeta $w.color $w.top $w.mid $w.bot -in $w \
		-side top -fill x
    }

    method updateFrames {} {
	set w .ui[modname]
	
	destroy $w.mid 
	destroy $w.bot

	frame $w.mid -relief groove -bd 2
	frame $w.bot -relief groove -bd 2
        set n "$this-c update"

	set vars [$this-c getVars]
	set varlist [split $vars]

	scale $w.mid.s1 -label "Isovalue" \
		-from [lindex $varlist 0] -to [lindex $varlist 1] \
		-resolution [lindex $varlist 2] \
		-variable $this-isovalue \
		-length 3c -orient horizontal -command $n

	scale $w.bot.s1 -label "Time Step" \
		-from 0 -to [lindex $varlist 3] \
		-variable $this-timevalue \
		-length 3c -orient horizontal -command $n

	scale $w.bot.s2 -label "Resolution" \
		-from 0 -to [lindex $varlist 4] \
		-variable $this-resolution \
		-length 3c -orient horizontal -command $n

	pack $w.mid.s1 -in $w.mid \
		-side top -fill x
	pack $w.bot.s1 $w.bot.s2 -in $w.bot \
		-side top -fill both
	pack $w.mid $w.bot -in $w \
		-side top -fill x

    }
    
}


