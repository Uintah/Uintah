#
#  LightWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename LightWidget ""}

itcl_class LightWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name LightWidget
    }
}
