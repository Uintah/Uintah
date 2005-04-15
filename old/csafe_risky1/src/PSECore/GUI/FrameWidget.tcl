#
#  FrameWidget.tcl
#
#  Written by:
#   James Purciful
#   Department of Computer Science
#   University of Utah
#   Apr. 1995
#
#  Copyright (C) 1995 SCI Group
#


catch {rename FrameWidget ""}

itcl_class FrameWidget {
    inherit BaseWidget
    constructor {config} {
	BaseWidget::constructor
	set name FrameWidget
    }

    method scale_changed {newscale} {
    }
}
