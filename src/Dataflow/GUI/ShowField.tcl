#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#  
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#  
#  The Original Source Code is SCIRun, released March 12, 2001.
#  
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
#  University of Utah. All Rights Reserved.
#

package require Iwidgets 3.0  

itcl_class SCIRun_Visualization_ShowField {
    inherit Module
    constructor {config} {
	set name ShowField
	set_defaults
    }

    method set_defaults {} {
	global $this-nodes-on
	global $this-nodes-as-disks
	global $this-edges-on
	global $this-faces-on
	global $this-vectors-on
	global $this-use-normals
	global $this-normalize_vectors
	global $this-node_display_type
	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	global $this-node_scale
	global $this-edge_scale
	global $this-vectors_scale
	global $this-resolution
	global $this-active_tab
	global $this-has_vec_data
	set $this-node_display_type Spheres
	set $this-edge_display_type Lines
	set $this-node_scale 0.03
	set $this-edge_scale 0.03
	set $this-vectors_scale 0.03
	set $this-def-color-r 0.5
	set $this-def-color-g 0.5
	set $this-def-color-b 0.5
	set $this-nodes-on 1
	set $this-nodes-as-disks 0
	set $this-edges-on 1
	set $this-faces-on 1
	set $this-vectors-on 0
	set $this-normalize_vectors 0
	set $this-resolution 4
	set $this-has_vec_data 0
	set $this-active_tab "Nodes"
	set $this-use-normals 0
	trace variable $this-active_tab w "$this switch_to_active_tab"
	trace variable $this-has_vec_data w "$this vec_tab_changed"
	trace variable $this-nodes-as-disks w "$this disk_render_status_changed"
    }

    method raiseColor {col color colMsg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     raise $window.color
	     return;
	 } else {
	     toplevel $window.color
	     makeColorPicker $window.color $color \
		     "$this setColor $col $color $colMsg" \
		     "destroy $window.color"
	 }
    }

    method setColor {col color colMsg} {
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]

	 set window .ui[modname]
	 $col config -background [format #%04x%04x%04x $ir $ig $ib]
	 $this-c $colMsg
    }

    method addColorSelection {frame color colMsg} {
	 #add node color picking 
	 global $color
	 global $color-r
	 global $color-g
	 global $color-b
	 set ir [expr int([set $color-r] * 65535)]
	 set ig [expr int([set $color-g] * 65535)]
	 set ib [expr int([set $color-b] * 65535)]
	 
	 frame $frame.colorFrame
	 frame $frame.colorFrame.col -relief ridge -borderwidth \
		 4 -height 0.8c -width 1.0c \
		 -background [format #%04x%04x%04x $ir $ig $ib]
	 
	 set cmmd "$this raiseColor $frame.colorFrame.col $color $colMsg"
	 button $frame.colorFrame.set_color \
		 -text "Default Color" -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left
	 pack $frame.colorFrame -side left

    }

    method set_active_tab {act} {
	global $this-active_tab
	#puts stdout $act
	set $this-active_tab $act
    }

    method switch_to_active_tab {name1 name2 op} {
	#puts stdout "switching"
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set dof [$window.options.disp.frame_title childsite]
	    $dof.tabs view [set $this-active_tab]
	}
    }

    # Nodes Tab
    method add_nodes_tab {dof} {
	
	set nodes [$dof.tabs add -label "Nodes" \
		-command "$this set_active_tab \"Nodes\""]
	
	checkbutton $nodes.show_nodes \
		-text "Show Nodes" \
		-command "$this-c toggle_display_nodes" \
		-variable $this-nodes-on

	global $this-node_display_type
	
	if {[set $this-nodes-as-disks] == 1} {
	    make_labeled_radio $nodes.radio \
		    "Node Display Type" "$this-c node_display_type" top \
		    $this-node_display_type \
		    {{Spheres Spheres} {Axes Axes} {Point Points} {Disks Disks}}
	} else {
	    make_labeled_radio $nodes.radio \
		    "Node Display Type" "$this-c node_display_type" top \
		    $this-node_display_type \
		    {{Spheres Spheres} {Axes Axes} {Point Points}}
	}

	pack $nodes.show_nodes $nodes.radio -fill y -anchor w

	expscale $nodes.slide -label NodeScale \
		-orient horizontal \
		-variable $this-node_scale -command "$this-c node_scale"

	bind $nodes.slide.scale <ButtonRelease> \
		"$this-c needexecute"
    }

    # Edges Tab
    method add_edges_tab {dof} {

	set edge [$dof.tabs add -label "Edges" \
		-command "$this set_active_tab \"Edges\""]
	checkbutton $edge.show_edges \
		-text "Show Edges" \
		-command "$this-c toggle_display_edges" \
		-variable $this-edges-on

	global $this-edge_display_type
	make_labeled_radio $edge.radio \
		"Edge Display Type" "$this-c edge_display_type" top \
		$this-edge_display_type {{Cylinders Cylinders} {Lines Lines}}

	pack $edge.show_edges $edge.radio \
		-side top -fill y -anchor w

	expscale $edge.slide -label CylinderScale \
		-orient horizontal \
		-variable $this-edge_scale -command "$this-c edge_scale"

	bind $edge.slide.scale <ButtonRelease> \
		"$this-c needexecute"
    }

    # Faces Tab
    method add_faces_tab {dof} {
	set face [$dof.tabs add -label "Faces" \
		-command "$this set_active_tab \"Faces\""]
	checkbutton $face.show_faces \
		-text "Show Faces" \
		-command "$this-c toggle_display_faces" \
		-variable $this-faces-on
	checkbutton $face.use_normals \
		-text "Use Face Normals" \
		-command "$this-c rerender_faces" \
		-variable $this-use-normals
	pack $face.show_faces $face.use_normals -side top -fill y -anchor w
    }


    # Vector Tab
    method add_vector_tab {dof} {

	set vector [$dof.tabs add -label "Vectors" \
		-command "$this set_active_tab \"Vectors\""]
	checkbutton $vector.show_vectors \
		-text "Show Vectors" \
		-command "$this-c toggle_display_vectors" \
		-variable $this-vectors-on

	checkbutton $vector.normalize_vectors \
		-text "Normalize Vectors before scaling" \
		-command "$this-c toggle_normalize" \
		-variable $this-normalize-vectors

#	 global $this-vector_display_type
#	 make_labeled_radio $vector.radio \
#		 "Vector Display Type" "$this-c vector_display_type" top \
#		 $this-vector_display_type {{Cylinders Cylinders} {Lines Lines}}
#
	pack $vector.show_vectors $vector.normalize_vectors \
		-side top -fill y -anchor w

	expscale $vector.slide -label CylinderScale \
		-orient horizontal \
		-variable $this-vectors_scale -command "$this-c data_scale"

	bind $vector.slide.scale <ButtonRelease> \
		"$this-c needexecute"
    }

    method disk_render_status_changed {name1 name2 op} {
	#puts stdout "called disk_render_status_changed"
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set dof [$window.options.disp.frame_title childsite]
	    $dof.tabs delete "Nodes"
	    add_nodes_tab $dof
	    $dof.tabs view [set $this-active_tab]
	}
    }

    method vec_tab_changed {name1 name2 op} {
	global $this-has_vec_data

	set window .ui[modname]
	if {[winfo exists $window]} {
	    set dof [$window.options.disp.frame_title childsite]	
	    if {[set $name1] == 1} { 
		add_vector_tab $dof
		$dof.tabs view [set $this-active_tab]
	    } else {
		$dof.tabs delete "Vectors"
	    }
	}
    }
    method ui {} {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    raise $window
	    return;
	}
	toplevel $window
	
	#frame for all options to live
	frame $window.options
 
	# node frame holds ui related to vert display (left side)
	frame $window.options.disp -borderwidth 2
	pack $window.options.disp -padx 2 -pady 2 -side left -fill y

	set n "$this-c needexecute"	

	# Display Options
	iwidgets::labeledframe $window.options.disp.frame_title \
		-labelpos nw -labeltext "Display Options"
	set dof [$window.options.disp.frame_title childsite]

	iwidgets::tabnotebook  $dof.tabs -height 250 -raiseselect true 
	#label $window.options.disp.frame_title -text "Display Options"

	add_nodes_tab $dof
	add_edges_tab $dof
	add_faces_tab $dof
	if {[set $this-has_vec_data] == 1} {
	    add_vector_tab $dof
	}

	global $this-active_tab
	# view the active tab
	$dof.tabs view [set $this-active_tab]	
	$dof.tabs configure -tabpos "n"

	pack $dof.tabs -side top -expand yes

	#pack notebook frame
	pack $window.options.disp.frame_title -side top -expand yes
	
	#add bottom frame for execute and dismiss buttons
	frame $window.control -relief groove -borderwidth 2
	frame $window.def_col -borderwidth 2

	addColorSelection $window.def_col $this-def-color \
		"default_color_change"


	# Cylinder and Sphere Resolution
	iwidgets::labeledframe $window.resolution \
		-labelpos nw -labeltext "Cylinder and Sphere Resolution"
	set res [$window.resolution childsite]

	scale $res.scale -orient horizontal -variable $this-resolution \
		-from 3 -to 20 -showvalue true -resolution 1

	pack $res.scale -side top -fill both -expand 1

	pack $window.options -padx 2 -pady 2 -side top
	pack $window.resolution -padx 2 -pady 2 -side top -fill x -expand 1
	pack $window.def_col $window.control -padx 2 -pady 2 -side top

	button $window.control.execute -text Execute -command $n
	button $window.control.dismiss -text Dismiss -command "destroy $window"
	pack $window.control.execute $window.control.dismiss -side left 
    }
}


















