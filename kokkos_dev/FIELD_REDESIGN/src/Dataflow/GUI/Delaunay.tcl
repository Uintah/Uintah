
catch {rename Delaunay ""}

itcl_class SCIRun_Mesh_Delaunay {
    inherit Module
    constructor {config} {
	set name Delaunay
	set_defaults
    }
    method set_defaults {} {
	global $this-nnodes
	set $this-nnodes 1

	global $this-cleanup
	set $this-cleanup 1
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w

	scale $w.maxnode -from 0 -to 1000 -orient horizontal \
		-length 400 -variable $this-nnodes
	make_labeled_radio $w.cleanup "Cleanup:" "" \
		left $this-cleanup \
		{{True 1} {False 0}}
	button $w.b -text "Execute" -command "$this-c needexecute"
	pack $w.maxnode -fill x
	pack $w.cleanup $w.b
    }
}

