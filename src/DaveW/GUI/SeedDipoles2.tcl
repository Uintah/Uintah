catch {rename DaveW_FEM_SeedDipoles2 ""}

itcl_class DaveW_FEM_SeedDipoles2 {
    inherit Module
    constructor {config} {
	set name SeedDipoles2
	set_defaults
    }
    method set_defaults {} {
	global $this-seedRandTCL
	global $this-numDipolesTCL
	global $this-dipoleMagnitudeTCL
	set $this-seedRandTCL 0
	set $this-numDipolesTCL 6
	set $this-dipoleMagnitudeTCL 10
    }
    method make_entry {w text v c} {
        frame $w
        label $w.l -text "$text"
        pack $w.l -side left
        entry $w.e -textvariable $v
        bind $w.e <Return> $c
        pack $w.e -side right
    }
    method ui {} {
	global $this-seedRandTCL
	global $this-numDipolesTCL
	global $this-dipoleMagnitudeTCL

	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	
	make_entry $w.seedRand "Random Seed:" $this-seedRandTCL \
		"$this-c needexecute"
	make_entry $w.numDipoles "Number of Dipoles:" $this-numDipolesTCL \
		"$this-c needexecute"
	make_entry $w.dipoleMagnitude "Dipole Magnitude:" \
		$this-dipoleMagnitudeTCL "$this-c needexecute"
	button $w.exec -text " Execute" -command "$this-c needexecute"
	pack $w.seedRand $w.numDipoles $w.dipoleMagnitude -side top -fill x
	pack $w.exec -side top
    }
}
