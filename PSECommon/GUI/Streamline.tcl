
catch {rename Streamline ""}

itcl_class PSECommon_Visualization_Streamline {
    inherit Module
    constructor {config} {
	set name Streamline
	set_defaults
    }
    method set_defaults {} {
	if {[array size sourcelist] == 0} {
	    newsource
	}
    }
    method ui {} {
	set w .ui[modname]
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	#
	# Setup toplevel window
	#
	toplevel $w
	wm minsize $w 100 100
	set n "$this-c needexecute "

	frame $w.c1
	make_sourcesel $w.c1.sources
	pack $w.c1.sources -side top -padx 2 -pady 2 -ipadx 2 -ipady 2 \
		-fill y -anchor nw

	pack $w.c1 -side left -fill y -anchor nw
	make_sourceinfo $w.c1.si
	pack $w.c1.si -side top -padx 2 -pady 2 -ipadx 2 -ipady 2 \
		-fill both -anchor nw
	make_markerinfo $w.c1.mi
	pack $w.c1.mi -side top -padx 2 -pady 2 -ipadx 2 -ipady 2 \
		-fill both -anchor nw

	frame $w.c2
	pack $w.c2 -side left -fill y -anchor nw
	make_intinfo $w.c2.ii
	pack $w.c2.ii -side top -padx 2 -pady 2 -ipadx 2 -ipady 2 \
		-fill both -anchor nw
	make_animinfo $w.c2.ai
	pack $w.c2.ai -side top -padx 2 -pady 2 -ipadx 2 -ipady 2 \
		-fill both -anchor nw

	$ss.list selection clear 0 end
	$ss.list selection set 0
	entry $w.c2.position
	bind $w.c2.position <Return> "$this reposition"
	pack $w.c2.position -side top -padx 2 -pady 2 -ipadx 2 -ipady 2 \
	    -fill x -anchor nw

	make_labeled_radio $w.c2.drawmode "Drawmode:" "$this-c needexecute" \
	    left $this-drawmode \
	    {{"long trace" 1} {"short trace" 2} {"fade trace" 3}}
	scale $w.c2.drawdist -from 0 -to 1.0 -resolution 0 -orient horizontal \
	    -label "Draw distance:"
	pack $w.c2.drawmode $w.c2.drawdist -side top -padx 2 -pady 2 \
	    -ipadx 2 -ipady 2 -fill x -anchor nw

	selectsource [$ss.list get 0]


	button $w.execute -text "Execute" -command "$this-c needexecute "
    }
    method reposition {} {
	$this-c reposition $currentsource
    }
    protected ss ""
    method make_sourcesel {c} {
	set ss $c
	#
	# Panel for Source selection
	#
	frame $ss -relief groove -borderwidth 2

	#
	# The list of sources
	#
	label $ss.label -text "Sources:"
	pack $ss.label -side top -anchor w
	listbox $ss.list -height 6 -width 20 \
		-yscroll "$c.scroll set" \
		-exportselection false
	bind $ss.list <Button-1> \
		"$this selectsource \[%W get \[%W nearest %y\]\]"
	scrollbar $ss.scroll -command "$c.list yview"
	pack $ss.list -side left -fill x
	pack $ss.scroll -side left -fill y
	rebuild_sourcelist

	button $ss.new -text "New" -command "$this newsource"
	button $ss.clone -text "Clone" -command "$this clonesource"
	button $ss.delete -text "Delete" -command "$this delsource"
# Someday we would like to do multiple selections, but that
# is a little ambitious for now...
#	button $ss.selectall -text "Select All" \
#		-command "$c.list selection set 0 end"
	pack $ss.new $ss.clone $ss.delete -side top -fill x
    }
    protected si
    method make_sourceinfo {c} {
	set si $c
	#
	# Source info
	#
	frame $si -relief groove -borderwidth 2

	label $si.label -text "Source Info:"
	pack $si.label -side top -anchor w

	puts "name sent to expscale = $si.widgetscale"
#	expscale $si.widgetscale -label "Widget Scale:" \
#		-orient horizontal
#	pack $si.widgetscale -fill x -side bottom

	#
	# Selector for source type - point, line, square, ring
	#
	make_labeled_radio $si.widgets "Source:" "$this sourcetype" \
		left $this-source \
		{Point Line Square Ring}
	pack $si.widgets -side bottom

	frame $si.f
	pack $si.f -side left -fill x
	label $si.f.label -text "Name:"
	pack $si.f.label -side left
	entry $si.f.name -width 20
	pack $si.f.name -fill x
	bind $si.f.name <Return> "$this changename \[$si.f.name get\]"

	button $si.find -text "Find Field"
	pack $si.find -side right

    }
    protected mi
    method make_markerinfo {c} {
	set mi $c
	#
	# Marker information
	#
	frame $mi -relief groove -borderwidth 2

	label $mi.label -text "Marker Info:"
	pack $mi.label -side top -anchor w

	#
	# Selector for marker type - LumiLine, line, tube, ribbon, surface, 
	#
	make_labeled_radio $mi.marker "Marker:" "$this selectmarker" \
		left $this-markertype \
		{LumLine Line Tube Ribbon Surface}
	pack $mi.marker -side top

	#
	# Parameters for different modes...
	#
	frame $mi.info
	pack $mi.info -side top -fill x

	frame $mi.info.iLine
	# later make this have a check...
	frame $mi.info.iLumLine 

	global $this-alphaval
	set $this-alphaval 1.0
	scale $mi.info.iLumLine.alpha -orient horizontal -label "Lumi ALpha" \
		-from 0.00 -to 1.0 -resolution 0.01 -variable "this-alphaval"
	checkbutton $mi.info.iLumLine.doseed -text "Use Scalar Seeds" 
		
	pack $mi.info.iLumLine.alpha $mi.info.iLumLine.doseed -anchor nw -fill x
	
	puts "$mi.info = $mi.info"
#	expscale $mi.info.iTube -orient horizontal -label "Tube scale:"
#	puts "$mi.info"
#	expscale $mi.info.iRibbon -orient horizontal -label "Ribbon scale:"
	scale $mi.info.iSurface -orient horizontal -from 0 -to 90 \
		-tickinterval 30 -label "Maximum Angle:"

	scale $mi.skip -from 1 -to 100 -orient horizontal \
		-label "Skip:"
	pack $mi.skip -side top -fill x

	make_labeled_radio $mi.colorize "Colorize?" "" \
		left $this-colorize \
		{Yes No}
	pack $mi.colorize -side top -fill x 
    }
    protected ii

    method set_algorithm {} {
	puts "Changing algorithm"
    }

    method make_intinfo {c} {
	set ii $c
	frame $ii -relief groove -borderwidth 2

	label $ii.label -text "Integration Info:"
	pack $ii.label -side top -anchor w

	#
	# Selector for algorithm - Exact, Euler, RK4, PC, Stream function
	#

	make_labeled_radio $ii.alg "Algorithm:" "$this set_algorithm" \
		left $this-algorithm \
		{Exact Euler RK4}
	pack $ii.alg -side top -fill x

	#
	# Parameters
#	expscale $ii.stepsize -label "Step size:" \
#		-orient horizontal
#	pack $ii.stepsize -fill x -pady 2
	
	scale $ii.maxsteps \
		-from 0 -to 10000 -label "Maximum steps:" \
		-showvalue true -tickinterval 1000 \
		-orient horizontal
	pack $ii.maxsteps -fill x -pady 2

	make_labeled_radio $ii.dir "Direction:" "" \
		left $this-dir \
		{Upstream Downstream Both}
	pack $ii.dir -side top -fill x

    }
    protected ai

    method make_animinfo {c} {
	set ai $c
	frame $ai -relief groove -borderwidth 2

	label $ai.label -text "Animation Info:"
	pack $ai.label -side top -anchor w

	#
	# Selector for animation type
	#
	make_labeled_radio $ai.anim "Animation:" "$this redo_animation" \
		left $this-animation \
		{None Time Position}
	pack $ai.anim -side top -fill x
	scale $ai.anim_steps -digits 3 \
		-from 1 -to 50 -label "N Steps:" \
		-showvalue true -tickinterval 10 \
		-orient horizontal
	pack $ai.anim_steps -side top -fill x
    }

    method need_find {} {
	global $this-need_find
	set $this-need_find 1
	$this-c needexecute
    }


    protected sourcelist
    protected nextsource 0

    method rebuild_sourcelist {} {
	if {$ss != ""} {
	    if {[winfo exists $ss.list]} {
		$ss.list delete 0 end
		set items [lsort [array names sourcelist]]
		foreach item $items {
		    $ss.list insert end $item
		}
	    }
	}
    }
    method redo_animation {} {
	$this-c needexecute
    }
	
    method newsource {} {
	set sid $nextsource
	incr nextsource
	$this-c newsource $sid
	if {$ss != ""} {
	    if {[winfo exists $ss.list]} {
		$ss.list insert end $sid
		$ss.list selection clear 0 end
		$ss.list selection set end
		$ss.list see end
	    }
	}
	set sourcelist($sid) $sid

	#
	# Setup default parameters for this source
	#
	set s $this-$sid
	global $s-source
	set $s-source "Line"

	global $s-markertype
	set $s-markertype "Tube"

	global $s-tubesize
	set $s-tubesize 2

	global $s-alphaval
	set $s-alphaval 1.0

	global $s-doseed
	set $s-doseed 1

	global $s-ribbonsize
	set $s-ribbonsize 4

	global $s-maxbend
	set $s-maxbend 20

	global $s-algorithm
	set $s-algorithm "RK4"

	global $s-animation
	set $s-animation "None"

	global $s-anim_timesteps
	set $s-anim_timesteps 30

	global $s-stepsize
	set $s-stepsize 0.3

	global $s-skip
	set $s-skip 25

	global $s-maxsteps
	set $s-maxsteps 1000

	global $s-widget_scale
	set $s-widget_scale 1

	global $s-direction
	set $s-direction Both

	global $s-position
	set $s-position ""

	global $s-drawmode
	set $s-drawmode 1

	global $s-drawdist
	set $s-drawdist .1
	
	$this-c need_find $sid
#	$this-c needexecute
    }
    method selectmarker {} {
	setup_markerspec $sourcelist($currentsource)
	$this-c needexecute
    }
    method setup_markerspec {sid} {
	set s $this-$sid
	#global $s-alphaval
	
	$mi.info.iLumLine.alpha config -variable $s-alphaval
	$mi.info.iLumLine.doseed config -variable $s-doseed
#	$mi.info.iTube config -variable $s-tubesize
#	$mi.info.iRibbon config -variable $s-ribbonsize
	$mi.info.iSurface config -variable $s-maxbend

	pack forget $mi.info.iLumLine
	pack forget $mi.info.iLine
#	pack forget $mi.info.iTube
#	pack forget $mi.info.iRibbon
	pack forget $mi.info.iSurface
	global $s-markertype
	set mode [set $s-markertype]
#	pack $mi.info.i$mode -fill x
    }
    protected currentsource
    method selectsource {sname} {
	set sid $sourcelist($sname)
	set currentsource $sname
	# Setup all of the UI components...

	set s $this-$sid

	if {$si == ""} {
	    return;
	}

	#
	# Connect all of the UI components to this source...
	#
#	$si.widgetscale config -variable $s-widget_scale
	change_radio_var $si.widgets $s-source
	$si.f.name delete 0 end
	$si.f.name insert 0 $sname

	change_radio_var $mi.marker $s-markertype
	setup_markerspec $sid
	$mi.skip config -variable $s-skip
	change_radio_var $mi.colorize $s-colorize

	change_radio_var $ii.alg $s-algorithm
	global $s-stepsize
#	$ii.stepsize config -variable $s-stepsize
	$ii.maxsteps config -variable $s-maxsteps
	change_radio_var $ii.dir $s-direction

	change_radio_var $ai.anim $s-animation
	$ai.anim_steps config -variable $s-animsteps
	set w .ui[modname]
	$w.c2.position config -textvariable $s-position
	change_radio_var $w.c2.drawmode $s-drawmode
	$w.c2.drawdist config -variable $s-drawdist
    }
    method changename {newname} {
	set sid $sourcelist($currentsource)
	unset sourcelist($currentsource)
	set sourcelist($newname) $sid
	set currentsource $newname
	rebuild_sourcelist
    }
    method sourcetype {} {
	# Called when changing the type of the source.
	# Update UI components...
	$this-c needexecute
    }
}
