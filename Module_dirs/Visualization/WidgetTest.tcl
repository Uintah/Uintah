
catch {rename WidgetTest ""}

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
	set $this-widget_type 6
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
		-digits 8 -variable $this-widget_scale -command "$this-c scale"
	pack $w.f.slide -in $w.f -side top -padx 2 -pady 2 -anchor w

	make_labeled_radio $w.f.wids "Widgets:" "$this-c select;$n" \
		top $this-widget_type \
		{{PointWidget 0} {ArrowWidget 1} \
		{CriticalPointWidget 2} \
		{CrossHairWidget 3} {GaugeWidget 4} \
		{RingWidget 5} {FrameWidget 6} \
		{ScaledFrameWidget 7} {BoxWidget 8} \
		{ScaledBoxWidget 9} {ViewWidget 10} \
		{PathWidget 11}}
	pack $w.f.wids

	button $w.f.nextmode -text "NextMode" -command "$this-c nextmode"
	pack $w.f.nextmode -fill x -pady 2
    }
}
