catch {rename DaveW_EGI_DipoleInSphere ""}

itcl_class DaveW_EGI_DipoleInSphere {
    inherit Module
    constructor {config} {
	set name DipoleInSphere
	set_defaults
    }
    method set_defaults {} {
	global $this-methodTCL
	set $this-method OneSphere
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w

	make_labeled_radio $w.filetype "Method:" "" top $this-methodTCL \
		{{OneSphere OneSphere}  \
		{ThreeSpheres ThreeSpheres} \
		{InfiniteMedium InfiniteMedium}}
	set $this-method OneSphere
	pack $w.filetype -side top -fill both
    }
}
