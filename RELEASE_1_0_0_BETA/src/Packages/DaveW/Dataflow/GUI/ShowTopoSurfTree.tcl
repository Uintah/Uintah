catch {rename TopoSurfTreeToGeom ""}

itcl_class DaveW_EEG_TopoSurfToGeom {
    inherit Module
    constructor {config} {
        set name TopoSurfToGeom
        set_defaults
    }

    method set_defaults {} {
	global $this-patchMode
	global $this-wireMode
	global $this-junctionMode
	global $this-nonjunctionMode
	global $this-rad
	set $this-patchMode patchTog
	set $this-wireMode wireTog
	set $this-junctionMode junctionTog
	set $this-nonjunctionMode nonjunctionTog
	set $this-rad 0
    }

    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
            raise $w
            return;
        }
        toplevel $w

	frame $w.f
	pack $w.f -side top -fill x -padx 2 -pady 2
	make_labeled_radio $w.patch "Patches" ""\
                top $this-patchMode \
                {{Separately patchSep}  \
		{Together patchTog} \
		{None patchNone}}
	make_labeled_radio $w.wire "Wires" ""\
                top $this-wireMode \
                {{Separately wireSep}  \
		{Together wireTog} \
		{None wireNone}}
	make_labeled_radio $w.junction "Junctions" ""\
                top $this-junctionMode \
                {{Separately junctionSep}  \
		{Together junctionTog} \
		{None junctionNone}}
	make_labeled_radio $w.nonjunction "Non-Junctions" ""\
                top $this-nonjunctionMode \
                {{Separately nonjunctionSep}  \
		{Together nonjunctionTog} \
		{None nonjunctionNone}}
	frame $w.sz
	label $w.sz.l -text "Junction sphere size: "
	entry $w.sz.r -width 7 -relief sunken -bd 2 -textvariable $this-rad
	pack $w.sz.l $w.sz.r -side left -padx 4 -fill x -expand 1
	button $w.b -text "Execute" -command "$this-c needexecute"
	pack $w.patch $w.wire $w.junction $w.nonjunction $w.sz $w.b -fill \
		both -expand 1
    }
}
