
itcl_class WidgetTest {
    inherit Module
    constructor {config} {
	set name WidgetTest
	set_defaults
    }
    method set_defaults {} {
	global $this-widget_scale
	set $this-widget_scale 0.01
	global $this-widget_type
	set $this-widget_type 5
    }
    method ui {} {
	set w .ui$this
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w
	frame $w.f
	pack $w.f -padx 2 -pady 2
	set n "$this-c needexecute"

	scale $w.f.slide -label Scale -from 0.001 -to 0.05 -length 6c \
		-showvalue true \
		-orient horizontal -resolution 0.001 \
		-digits 8 -variable $this-widget_scale -command $n
	pack $w.f.slide -in $w.f -side top -padx 2 -pady 2 -anchor w

	make_labeled_radio $w.f.wids "Widgets:" $n top $this-widget_type \
		{ {PointWidget 0} {ArrowWidget 1} \
		{CrossHairWidget 2} {GuageWidget 3} \
		{RingWidget 4} {FixedFrameWidget 5} {FrameWidget 6} \
		{ScaledFrameWidget 7} {SquareWidget 8} \
		{ScaledSquareWidget 9} {BoxWidget 10} \
		{ScaledBoxWidget 11} {CubeWidget 12} \
		{ScaledCubeWidget 13} }
	pack $w.f.wids
    }
}
