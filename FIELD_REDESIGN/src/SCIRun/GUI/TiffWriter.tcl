
catch {rename TiffWriter ""}

itcl_class SCIRun_Image_TiffWriter {
    inherit Module
    constructor {config} {
	set name TiffWriter
	set_defaults
    }
    method set_defaults {} {
      	global $this-resolution
	set $this-resolution 8bit

	$this-c needexecute
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	makeFilebox $w $this-filename "$this-c needexecute" "destroy $w"

#	frame $w.f
#	pack $w.f -padx 2 -pady 2 -fill x

	make_labeled_radio $w.resolution "Resolution:" "" left $this-resolution \
		{8bit 16bit RGB Scale8bit Scale16bit ScaleRGB}
	pack $w.resolution -side top
	entry $w.reso -textvariable $this-resolution -width 40 \
		-borderwidth 2 -relief sunken

    }
}
