
itcl_class Uintah_MPMViz_GridLines {
    inherit Module

    constructor {config} {
	set name GridLines
	set_defaults
    }

    method set_defaults {} {
	global $this-mode
	global $this-rad
	global $this-lineRep
	global $this-textSpace
	global $this-dim
	global $this-plane
	global $this-planeloc
	set $this-mode 0
	set $this-rad 0
	set $this-lineRep 0
	set $this-textSpace 10
	set $this-plane 0
	set $this-dim 0
	set $this-planeloc 0.5
    }
    method ui {} {
	global $this-rad
	global $this-lineRep
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	set n "$this-c needexecute"

	frame $w.m 
	frame $w.f -relief groove -borderwidth 2
	frame $w.d -relief groove -borderwidth 2
	pack $w.m $w.d $w.f -padx 2 -pady 2 -fill x -expand yes

	make_labeled_radio $w.m.r "Grid Type" $n left \
		$this-mode {{"None" 0} {"Inside" 1 } {"Outside" 2} {"Both" 3}}
	pack $w.m.r

	make_labeled_radio $w.d.r "2D\/3D" $n left $this-dim {{2D 0} {3D 1}}
	
	make_labeled_radio $w.d.p "Choose Plane" $n left \
	    $this-plane  {{ XY 0 } { XZ 1 } { YZ 2 }}
	scale $w.d.s -label "Plane Location" -orient horizontal \
	    -from 0 -to 1 -resolution 0.0001 \
	    -length 8c -variable $this-planeLoc -command $n

	pack $w.d.r $w.d.p $w.d.s -padx 2 -pady 2


	make_labeled_radio $w.f.shaft "Line draw style:" $n \
	    left $this-lineRep {{Lines 0} {Cylinders 1}}


	scale $w.f.rad -label "Size of Grid Lines" -orient horizontal \
	    -from 0 -to 1 -resolution 0.0001 \
	    -length 8c -variable $this-rad -command $n

	pack $w.f.shaft $w.f.rad -side top

	button $w.b -text Close -command "destroy $w"
	pack $w.b -side bottom -expand yes -fill x -padx 2 -pady 2
	

    }
}

