##
 #  LookupSplitSurface.tcl: General surface interpolation module
 #
 #  Written by:
 #   David Weinstein
 #   Department of Computer Science
 #   University of Utah
 #   Jully 1997
 #
 #  Copyright (C) 1997 SCI Group
 # 
 #  Log Information:
 #
 #  $Log$
 #  Revision 1.1  2000/10/29 04:34:46  dmw
 #  BuildFEMatrix -- ground an arbitrary node
 #  SolveMatrix -- when preconditioning, be careful with 0's on diagonal
 #  MeshReader -- build the grid when reading
 #  SurfToGeom -- support node normals
 #  IsoSurface -- fixed tet mesh bug
 #  MatrixWriter -- support split file (header + raw data)
 #
 #  LookupSplitSurface -- split a surface across a place and lookup values
 #  LookupSurface -- find surface nodes in a sfug and copy values
 #  Current -- compute the current of a potential field (- grad sigma phi)
 #  LocalMinMax -- look find local min max points in a scalar field
 #
 #
 ##

itcl_class PSECommon_Surface_LookupSplitSurface {
    inherit Module
    constructor {config} {
        set name LookupSplitSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-splitDirTCL
	global $this-splitValTCL
	set $this-splitDirTCL X
	set $this-splitValTCL 0
    }
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            raise $w
            return;
        }
	
        toplevel $w
        wm minsize $w 100 30
        frame $w.f
        set n "$this-c needexecute "
	global $this-splitDirTCL
        make_labeled_radio $w.splitdir "Split Direction:" "" left \
		$this-splitDirTCL {X Y Z}
        global $this-splitValTCL
        expscale $w.splitval -orient horizontal -label "Split value:" \
                -variable $this-splitValTCL
	button $w.e -text "Execute" -command $n
	pack $w.splitdir $w.splitval $w.e -side top
	pack $w.f
    }
}
