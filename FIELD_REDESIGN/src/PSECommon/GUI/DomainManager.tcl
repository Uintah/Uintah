itcl_class PSECommon_Domain_DomainManager {
    inherit Module

    constructor {config} {
	set name DomainManager
	set_defaults
    }

    method set_defaults {} {
    }

    #create the geom and attrib list boxes
    method make_list {iw text v m} {
	frame $iw -height 200 -width 150
	label $iw.label -text $text
	pack $iw.label -side top -side top
	listbox $iw.list -yscrollcommand "$iw.scroll set"
	pack $iw.list -side left -fill y
	scrollbar $iw.scroll -command "$iw.list yview"
	pack $iw.scroll -side right -fill y

    }

    method make_box {iw} {
	frame $iw -height 150 -width 150
	text $iw.text -relief sunken -bd 2 -yscrollcommand "$iw.scroll set" \
		-height 8 -width 20
	pack $iw.text -side left -fill y
	scrollbar $iw.scroll -command "$iw.list yview"
	pack $iw.scroll -side right -fill y
    }

    method load {} {
    }

    method save_selected {} {
    }

    method save_domain {} {
    }

    method remove {} {
    }

    # updates the geometry list on the UI
    method update_geom {args} {
	global w
	
	set args [join $args]
	$w.main.lists.geoms.list delete 0 [$w.main.lists.geoms.list size]
	$w.main.lists.geoms.list insert end [split $args ,]
    }

    # updates the attribute list on the UI
    method update_attrib {args} {
	global w
	set args [join $args]
	$w.main.lists.attribs.list delete 0 [$w.main.lists.attribs.list size]
	$w.main.lists.attribs.list insert end [split $args ,]
    }

    # stores a list of strings describing each geometry
    method update_geom_info {args} {
	global geom_info

	set geom_info [split $args ,]
	
    }

    method update_attrib_info {args} {
    }

    method ui {} {
	global this-savetype
	global w
	global geom_info
	global attrib_info

	set w .ui[modname]
	set n "$this-c needexecute "
	
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}
	toplevel $w 
	frame $w.main -width 350 -height 500

### Create the save and load frame and buttons
	frame $w.main.saveload
	set current $w.main.saveload
	button $current.lbutton -text "Load" -command "$this load"
	button $current.ssbutton -text "Save Selected" -command "$this save_selected"
	button $current.sdbutton -text "Save Domain" -command "$this save_domain"
	button $current.rbutton -text "Remove" -command "$this remove"
	pack $current.lbutton $current.ssbutton $current.sdbutton \
		$current.rbutton -side left -padx 5
	pack $current -side left -side top -fill x -padx 5 -pady 8

### Create the Geometries and Attributes listboxes
	frame $w.main.lists
	set current $w.main.lists
	make_list $current.geoms "Geometries:" $this-geoms $n
	make_list $current.attribs "Attributes:" $this-attribs $n
	pack $current.geoms -side left
	pack $current.attribs -side right
	pack $current -side top -fill x -padx 5 -pady 5 

### Create the Info boxes
	frame $w.main.info
	set current $w.main.info
	make_box $current.geom_info 
	make_box $current.attrib_info
	pack $current.geom_info -side left
	pack $current.attrib_info -side right
	pack $current -side top -fill x -padx 5 -pady 5

### Set up bindings for listboxes to update info boxes
	bind $w.main.lists.geoms.list <Button-1> {
	    # delete everything in the info box
	    $w.main.info.geom_info.text delete 1.0 end
	    set ind [$w.main.lists.geoms.list curselection]
	    if { [string length $ind] > 0 } {
		$w.main.info.geom_info.text insert end \
			[lindex $geom_info $ind]
	    }
	}

### Pack the main window
	pack $w.main -side top -side left
    }
}
