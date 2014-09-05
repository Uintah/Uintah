
catch {rename BoundGrid ""}

itcl_class BoundGrid {
    inherit Module
    constructor {config} {
	set name BoundGrid
	set_defaults
    }
    method set_defaults {} {
	global $this-use_lines

	set $this-use_lines 1
	puts "set_defaults"
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -expand 1 -fill x
	set n "$this-c needexecute "
	global $this-use_lines
	checkbutton $w.f.dointerp -text "Use Lines" -variable $this-use_lines \
		-command $n
	pack $w.f.dointerp -side top -fill x
    }
}
