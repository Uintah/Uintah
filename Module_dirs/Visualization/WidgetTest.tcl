
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

	global $this-widget_material
	initMaterial $this-widget_material

	$this-c select
	$this-c material
	$this-c needexecute
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
		{LightWidget 11} {PathWidget 12}}
	pack $w.f.wids

	button $w.f.nextmode -text "NextMode" -command "$this-c nextmode"
	pack $w.f.nextmode -fill x -pady 2
	
	toplevel $w.mat
	makeMaterialEditor $w.mat $this-widget_material "$this-c material" "destroy $w.mat"
    }
}

proc initColor {c r g b} {
    global $c-r $c-g $c-b
    set $c-r $r
    set $c-g $g
    set $c-b $b
}

proc initMaterial {matter} {
    initColor $matter-ambient 0.1 0.2 0.3
    initColor $matter-diffuse 0.4 0.5 0.6
    initColor $matter-specular 0.7 0.8 0.9
    global $matter-shininess
    set $matter-shininess 10.0
    initColor $matter-emission 0.1 0.4 0.7
    global $matter-reflectivity
    set $matter-reflectivity 0.5
    global $matter-transparency
    set $matter-transparency 0
    global $matter-refraction_index
    set $matter-refraction_index 1.0
}

source $sci_root/TCL/MaterialEditor.tcl
