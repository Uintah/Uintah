#
#  ArrowWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename ArrowWidget ""}

itcl_class ArrowWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name ArrowWidget
    }
}
