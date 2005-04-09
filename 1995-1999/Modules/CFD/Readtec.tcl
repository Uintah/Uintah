
catch {rename Readtec ""}

itcl_class Readtec {
    inherit Module
    constructor {config} {
	set name Readtec
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
	frame $w.f1 -relief groove -bd 2
	frame $w.f2 -relief groove -bd 2
	frame $w.f2.top -relief groove -bd 2
	frame $w.f2.mid -relief groove -bd 2
	frame $w.f2.bot -relief groove -bd 2

        set n "$this-c needexecute "
        make_labeled_radio $w.f2.top.var "Scalar Variable" $n top \
		$this-svar {{P 0} {T 1} {E 2} {RHO 3} {CKS 4} {THE 5}}

        make_labeled_radio $w.f2.top.flu "Fluid #" $n top \
		$this-sfluid {{"Fluid 1" 1} {"Fluid 2" 2} \
		{"Fluid 3" 3} {"Fluid 4" 4}}

	make_labeled_radio $w.f2.mid.var "Vector Variable" $n top \
		$this-vvar {{"U,V" 0} {"MOX,MOY" 1}}

        make_labeled_radio $w.f2.mid.flu "Fluid #" $n top \
		$this-vfluid {{"Fluid 1" 1} {"Fluid 2" 2} \
		{"Fluid 3" 3} {"Fluid 4" 4}}
	
	make_labeled_radio $w.f2.bot.flu "Fluid Particles" $n top \
		$this-pfluid { {"Material 1" 1} {"Material 2" 2} \
		{"Material 3" 3} {"Material 4" 4} }

	make_labeled_radio  $w.f2.bot.var "Scalar Variable" $n top \
		$this-pvar {{P 0} {T 1} {E 2} {RHO 3} {CKS 4} {THE 5}}
	
	pack $w.f2.top.var $w.f2.top.flu -in $w.f2.top -side left -fill both
	pack $w.f2.mid.var $w.f2.mid.flu -in $w.f2.mid -side left -fill both
	pack $w.f2.bot.flu $w.f2.bot.var -in $w.f2.bot -side left -fill both
	pack $w.f2.top $w.f2.mid $w.f2.bot -in $w.f2

	pack $w.f1 $w.f2 -in $w -side left

	makeFilebox $w.f1 $this-filebase "$this-c needexecute" "destroy $w"
    }
}
