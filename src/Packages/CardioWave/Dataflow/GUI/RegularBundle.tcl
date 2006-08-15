itcl_class CardioWave_TissueModel_RegularBundle {
    inherit Module
    constructor {config} {
        set name RegularBundle
        set_defaults
    }

    method set_defaults {} {
       global $this-cells-x
       global $this-cells-y
       global $this-cells-z
       global $this-elems-x-ics
       global $this-elems-y-ics
       global $this-elems-x-ecs
       global $this-elems-y-ecs
       global $this-elems-z
       global $this-bath-start
       global $this-bath-end
       
       global $this-cell-length
       global $this-cell-crosssection
       global $this-ics-vol-frac

       global $this-lateral-connection-x
       global $this-lateral-connection-y

       set $this-cells-x 3
       set $this-cells-y 3
       set $this-cells-z 3
       set $this-elems-x-ics 4
       set $this-elems-y-ics 4
       set $this-elems-x-ecs 2
       set $this-elems-y-ecs 2
       set $this-elems-z 10
       set $this-bath-start 6
       set $this-bath-end 6
       
       set $this-cell-length 100E-6
       set $this-cell-crosssection 300E-12
       set $this-ics-vol-frac 0.8

       set $this-lateral-connection-x 10
       set $this-lateral-connection-y 10
             
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
        
        frame $w.f
        pack $w.f -expand yes -fill x
        
        label $w.f.l1 -text "Number of cells in X direction"
        label $w.f.l2 -text "Number of cells in Y direction"
        label $w.f.l3 -text "Number of cells in Z direction"
        
        label $w.f.l4 -text "Number of elements in ICS in X direction"
        label $w.f.l5 -text "Number of elements in ICS in Y direction"
        label $w.f.l6 -text "Number of elements in ECS in X direction"
        label $w.f.l7 -text "Number of elements in ECS in Y direction"
        label $w.f.l8 -text "Number of elements in Z direction"
        
        label $w.f.l9 -text "Number of elements in bath at start"
        label $w.f.l10 -text "Number of elements in bath at end"

        label $w.f.l11 -text "Cell length"
        label $w.f.l12 -text "Cell cross section"
        label $w.f.l13 -text "ICS volume fraction"

        label $w.f.l14 -text "Number of elements for lateral connection in X"
        label $w.f.l15 -text "Number of elements for lateral connection in Y"

        entry $w.f.e1 -textvariable $this-cells-x
        entry $w.f.e2 -textvariable $this-cells-y
        entry $w.f.e3 -textvariable $this-cells-z
        entry $w.f.e4 -textvariable $this-elems-x-ics
        entry $w.f.e5 -textvariable $this-elems-y-ics
        entry $w.f.e6 -textvariable $this-elems-x-ecs
        entry $w.f.e7 -textvariable $this-elems-y-ecs
        entry $w.f.e8 -textvariable $this-elems-z

        entry $w.f.e9 -textvariable $this-bath-start
        entry $w.f.e10 -textvariable $this-bath-end

        entry $w.f.e11 -textvariable $this-cell-length
        entry $w.f.e12 -textvariable $this-cell-crosssection
        entry $w.f.e13 -textvariable $this-ics-vol-frac

        entry $w.f.e14 -textvariable $this-lateral-connection-x
        entry $w.f.e15 -textvariable $this-lateral-connection-y

        grid $w.f.l1 -row 0 -column 0 -sticky e
        grid $w.f.l2 -row 1 -column 0 -sticky e
        grid $w.f.l3 -row 2 -column 0 -sticky e
        grid $w.f.l4 -row 3 -column 0 -sticky e
        grid $w.f.l5 -row 4 -column 0 -sticky e
        grid $w.f.l6 -row 5 -column 0 -sticky e
        grid $w.f.l7 -row 6 -column 0 -sticky e
        grid $w.f.l8 -row 7 -column 0 -sticky e
        grid $w.f.l9 -row 8 -column 0 -sticky e
        grid $w.f.l10 -row 9 -column 0 -sticky e
        grid $w.f.l11 -row 10 -column 0 -sticky e
        grid $w.f.l12 -row 11 -column 0 -sticky e
        grid $w.f.l13 -row 12 -column 0 -sticky e
        grid $w.f.l14 -row 13 -column 0 -sticky e
        grid $w.f.l15 -row 14 -column 0 -sticky e

        grid $w.f.e1 -row 0 -column 1 -sticky news
        grid $w.f.e2 -row 1 -column 1 -sticky news
        grid $w.f.e3 -row 2 -column 1 -sticky news
        grid $w.f.e4 -row 3 -column 1 -sticky news
        grid $w.f.e5 -row 4 -column 1 -sticky news
        grid $w.f.e6 -row 5 -column 1 -sticky news
        grid $w.f.e7 -row 6 -column 1 -sticky news
        grid $w.f.e8 -row 7 -column 1 -sticky news
        grid $w.f.e9 -row 8 -column 1 -sticky news
        grid $w.f.e10 -row 9 -column 1 -sticky news
        grid $w.f.e11 -row 10 -column 1 -sticky news
        grid $w.f.e12 -row 11 -column 1 -sticky news
        grid $w.f.e13 -row 12 -column 1 -sticky news
        grid $w.f.e14 -row 13 -column 1 -sticky news
        grid $w.f.e15 -row 14 -column 1 -sticky news

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }
}


