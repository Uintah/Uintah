#
#  RingWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename RingWidget ""}

itcl_class RingWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name RingWidget
    }

    method scale_changed {newscale} {
    }
}
