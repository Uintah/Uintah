
catch {rename SoundInput ""}

itcl_class SoundInput {
    inherit Module
    constructor {config} {
	set name SoundInput
	set_defaults
    }
    method set_defaults {} {
	global $this-onoff
	set $this-onoff 0
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	checkbutton $w.onoff -text "Sound On/Off" \
		-variable $this-onoff -command "$this do_switch"
	pack $w.onoff
    }
    method do_switch {} {
	global $this-onoff
	if {[set $this-onoff]} {
	    # Turning it on...
	    $this-c needexecute
	} else {
	    # The module will take care of the rest
	}
    }
}
