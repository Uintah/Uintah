##
 #  LookupSurface.tcl: General surface interpolation module
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
 #  Revision 1.1.2.1  2000/10/31 02:19:19  dmw
 #  Merging PSECommon changes from HEAD to FIELD_REDESIGN branch
 #
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

itcl_class PSECommon_Surface_LookupSurface {
    inherit Module
    constructor {config} {
        set name LookupSurface
        set_defaults
    }
    method set_defaults {} {
	global $this-constElemsTCL
	set $this-constElemsTCL 0
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
	global $this-constElemsTCL
	checkbutton $w.f.b -text "Constant Elements" \
		-variable $this-constElemsTCL
	button $w.f.e -text "Execute" -command $n
	pack $w.f.b $w.f.e -side top
	pack $w.f
    }
}
