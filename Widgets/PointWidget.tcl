#
#  PointWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename PointWidget ""}

itcl_class PointWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name PointWidget
    }
}
