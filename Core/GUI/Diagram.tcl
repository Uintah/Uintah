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

#  Diagram.tcl
#  Written by:
#   Yarden Livnat
#   Department of Computer Science
#   University of Utah
#   July 2001
#  Copyright (C) 2001 SCI Group

package require Iwidgets 3.0

class Diagram {

    variable w 
    set val(0) 1

    constructor { args } {
    }

    destructor {
	delete object  $w.diagram
    }

    method ui { window name } {
	set w $window

	iwidgets::labeledframe $w.diagram -labeltext $name  -labelpos nw
	pack $w.diagram -side left 
    }

    method add { n name } {
	set cs [$w.diagram childsite]
	checkbutton $cs.$name -text $name -variable val($name) \
	    -command "$this-c select $n \$val($name)"
	$cs.$name select
	pack $cs.$name -side left
    }
}
