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


package require Iwidgets 3.1

class ItPDSimPartGui {

    variable w
    variable n

    constructor {} {
	set n 0
    }

    method ui { window } {

	puts "ItPDSimPartGui UI"
	set w $window

        iwidgets::entryfield $w.df -labeltext "df:" \
	    -command "$this-c df \[$w.df get\]" 

#-validate numeric 

	frame $w.children

	pack $w.df  -anchor w
	pack $w.children

    }

    method set-df { df } {
	if { [$w.df get] != $df } {
	    $w.df set $df
	}
    }

    method new-child-window { name } {
	set child [iwidgets::Labeledframe $w.children.$n -labeltext $name]
	pack $child -side top -anchor w
	incr n
	return [$child childsite]
    }
}


