##
 #  SiReInput.tcl: Read in the k-space data for SiRe
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Aug 1996
 #  Copyright (C) 1996 SCI Group
 ##

catch {rename DaveW_SiRe_SiReInput ""}

itcl_class DaveW_SiRe_SiReInput {
    inherit Module
    constructor {config} {
        set name SiReInput
        set_defaults
    }
    method set_defaults {} {
	global $this-PFileStr
	global $this-NPasses
	global $this-ShrinkFactor
	set $this-PFileStr P59904
	set $this-NPasses 1
	set $this-ShrinkFactor 4
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }

        toplevel $w
        wm minsize $w 300 80
        frame $w.f
        pack $w.f -padx 2 -pady 2 -side top -expand yes
        set n "$this-c needexecute "
        global $this-PFileStr
	global $this-NPasses
	global $this-ShrinkFactor
	make_labeled_radio $w.f.fpf "Series to reconstruct: " " " \
		left $this-PFileStr \
		{{"Phantom Part" P59904} \
		{"Carotid Arteries" P20992}}
	scale $w.f.sh -label "Shrink Factor (log)" \
		-variable $this-ShrinkFactor -orient horizontal -from 0 \
		-to 8 -showvalue true
	scale $w.f.np -label "Number of Passes" \
		-variable $this-NPasses -orient horizontal -from 1 \
		-to 4 -showvalue true
	pack $w.f.fpf $w.f.sh $w.f.np -side top -expand 1 -fill x
	pack $w.f -expand 1 -fill x
    }
}
