##
 #  DipoleInAnisoSpheres.tcl:
 #
 #  Author: Sascha Moehrs
 #
 ##

package require Iwidgets 3.0

catch {rename DipoleInAnisoSpheres ""}

itcl_class BioPSE_Forward_DipoleInAnisoSpheres {
    inherit Module
    constructor {config} {
        set name DipoleInAnisoSpheres
        set_defaults
    }
    method set_defaults {} {

		# accuracy of the series expansion / max expansion terms
		global $this-accuracy
		set $this-accuracy 0.00001
		global $this-expTerms
		set $this-expTerms 100

    }
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return
        }

        toplevel $w
		# wm minsize $w 300 100

		frame $w.f 
		pack $w.f -padx 2 -pady 2 -expand 1 -fill x

		# accuracy / expansion terms
		iwidgets::labeledframe $w.f.a -labelpos "nw" -labeltext "series expansion"
		set ac [$w.f.a childsite]

		global $this-accuracy
		label $ac.la -text "accuracy: "
		entry $ac.ea -width 20 -textvariable $this-accuracy
		bind  $ac.ea <Return> "$this-c needexecute"
		grid  $ac.la -row 0 -column 0 -sticky e
		grid  $ac.ea -row 0 -column 1 -columnspan 2 -sticky "ew"
		
		global $this-expTerms
		label  $ac.le -text "expansion terms: "
		entry  $ac.ee -width 20 -textvariable $this-expTerms -state disabled
		# bind   $ac.ee <Return> "$this-c needexecute"
		grid   $ac.le -row 1 -column 0 -sticky e
		grid   $ac.ee -row 1 -column 1 -columnspan 2 -sticky "ew"

		grid columnconfigure . 1 -weight 1

		pack $w.f.a -side top -fill x -expand 1

    }
}
