##
 #  MatrixSend.tcl: Send matrix to a host:port
 #  Written by:
 #   Oleg
 #   Department of Computer Science
 #   University of Utah
 #   01Jan05
 ##

catch {rename Oleg_DataIO_MatrixSend ""}

itcl_class Oleg_DataIO_MatrixSend {
    inherit Module
    constructor {config} {
        set name MatrixSend
        set_defaults
    }
    method set_defaults {} {
	global $this-hpTCL
	set $this-hpTCL ""
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-hpTCL
	frame $w.f.hp
	label $w.f.hp.l -text "host:port "
	entry $w.f.hp.e -relief sunken -width 21 -textvariable $this-hpTCL
	pack $w.f.hp.l $w.f.hp.e -side left
	pack $w.f.hp -side top
        pack $w.f -side top -expand yes
    }
}
