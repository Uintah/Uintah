
catch {rename FieldRGAug ""}

itcl_class FieldRGAug {
    inherit Module
    constructor {config} {
	set name FieldRGAug
	set_defaults
    }
    method set_defaults {} {
	global $this-nx
	global $this-ny
	global $this-nz

	set $this-nx 151
	set $this-ny 151
	set $this-nz 31

    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	
	frame $w.f

	pack $w.f -fill x

	scale $w.f.nx -label "Number in X" \
		-orient horizontal \
		-from 1 -to 300 \
		-variable $this-nx
	scale $w.f.ny -label "Number in Y" \
		-orient horizontal \
		-from 1 -to 300 \
		-variable $this-ny
	scale $w.f.nz -label "Number in Z" \
		-orient horizontal \
		-from 1 -to 300 \
		-variable $this-nz
	pack $w.f.nx $w.f.ny $w.f.nz -anchor nw -fill x

	button $w.f.file -text "File Thing" -command "$this fbox"

	pack $w.f.file
    }
    method fbox {} {
	set w .ui$this-file
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	makeFilebox $w $this-filename "$this-c needexecute" "destroy $w"
    }
}
