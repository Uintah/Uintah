#
#  ScaledBoxWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename ScaledBoxWidget ""}

itcl_class ScaledBoxWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name ScaledBoxWidget
    }
}
