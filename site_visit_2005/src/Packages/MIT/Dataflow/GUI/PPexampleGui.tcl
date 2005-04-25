#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#


package require Iwidgets 3.0

itcl::class PPexampleGui {

    variable w
    variable n

    constructor {} {
    }

    method ui { window } {

	puts "PPexampleGui UI"
	set w $window

        iwidgets::entryfield $w.burning -labeltext "Burning:" \
	    -validate numeric -command "$this-c burning \[$w.burning get\]" 

        iwidgets::entryfield $w.monitor -labeltext "Monitor:" \
	    -validate numeric -command "$this-c monitor \[$w.monitor get\]" 

        iwidgets::entryfield $w.thin -labeltext "Thin:" \
	    -validate numeric -command "$this-c thin  \[$w.thin get\]" 

        iwidgets::entryfield $w.kappa -labeltext "Kappa:" \
	    -validate numeric -command "$this-c kappa \[$w.kappa get\]" 

	button $w.exec -text "Execute" -command "$this-c exec"

	frame $w.children


	pack $w.burning $w.monitor $w.thin  -anchor w
	pack $w.children
	pack $w.exec -anchor w

	set n 0
	#iwidgets::Labeledframe $w.graph -labeltext "Progress"
	#pack $w.graph -expand yes -fill both
    }

    method new-child-window { name } {
#	set child [frame $w.children.$n]
#	pack $child -side top
#	incr n
#	return $child
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w
	incr n
	return [$child childsite]
    }
}













