
catch {rename ParticleSetReader ""}

itcl_class ParticleSetReader {
    inherit Module
    constructor {config} {
	set name ParticleSetReader
	set_defaults
    }
    method set_defaults {} {
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	makeFilebox $w $this-filename "$this-c needexecute" "destroy $w"
    }
}
