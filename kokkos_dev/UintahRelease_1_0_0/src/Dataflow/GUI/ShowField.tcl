#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
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
	global $this-nodes-transparency
	global $this-nodes-as-disks
	global $this-edges-on
	global $this-faces-on
	global $this-vectors-on
	global $this-tensors-on
	global $this-scalars-on
	global $this-text-on
	global $this-use-normals
	global $this-edges-transparency
	global $this-scalars-transparency
	global $this-use-transparency
	global $this-normalize_vectors
	global $this-node_display_type
	global $this-edge_display_type
	global $this-data_display_type
	global $this-tensor_display_type
	global $this-scalar_display_type
	global $this-def-color-r
	global $this-def-color-g
	global $this-def-color-b
	global $this-def-color-a
	global $this-node_scale
	global $this-node_scaleNV
	global $this-edge_scale
	global $this-edge_scaleNV
	global $this-vectors_scale
	global $this-vectors_scaleNV
	global $this-tensors_scale
	global $this-tensors_scaleNV
	global $this-scalars_scale
	global $this-scalars_scaleNV
	global $this-node-resolution
	global $this-edge-resolution
	global $this-data-resolution
	global $this-active_tab
	global $this-has_vector_data
	global $this-has_tensor_data
	global $this-has_scalar_data
	global $this-interactive_mode
	global $this-bidirectional
	global $this-nodes-usedefcolor
	global $this-edges-usedefcolor
	global $this-faces-usedefcolor
	global $this-scalars-usedefcolor
	global $this-vectors-usedefcolor
	global $this-tensors-usedefcolor
	global $this-text-use-default-color
	global $this-text-color-r
	global $this-text-color-g
	global $this-text-color-b
	global $this-text-backface-cull
	global $this-text-fontsize
	global $this-text-precision
	global $this-text-render_locations
	global $this-text-show-data
	global $this-text-show-nodes
	global $this-text-show-edges
	global $this-text-show-faces
	global $this-text-show-cells
	global $this-field-name
	global $this-field-name-override
	global $this-field-name-update
	global $this-tensors-emphasis
	global $this-approx-div
	set $this-node_display_type Points
	set $this-edge_display_type Lines
	set $this-data_display_type Arrows
	set $this-tensor_display_type Boxes
	set $this-scalar_display_type Points
	set $this-node_scale 0.03
	set $this-node_scaleNV 0.03
	set $this-edge_scale 0.015
	set $this-edge_scaleNV 0.015
	set $this-vectors_scale 0.30
	set $this-vectors_scaleNV 0.30
	set $this-tensors_scale 0.30
	set $this-tensors_scaleNV 0.30
	set $this-scalars_scale 0.30
	set $this-scalars_scaleNV 0.30
	set $this-def-color-r 0.5
	set $this-def-color-g 0.5
	set $this-def-color-b 0.5
	set $this-def-color-a 0.5
	set $this-nodes-on 1
	set $this-nodes-transparency 0
	set $this-nodes-as-disks 0
	set $this-edges-on 1
	set $this-faces-on 1
	set $this-text-on 0
	set $this-vectors-on 0
	set $this-tensors-on 0
	set $this-scalars-on 0
	set $this-normalize_vectors 0
	set $this-node-resolution 6
	set $this-edge-resolution 6
	set $this-data-resolution 6
	set $this-has_vector_data 0
	set $this-has_tensor_data 0
	set $this-has_scalar_data 0
	set $this-active_tab "Nodes"
	set $this-use-normals 0
	set $this-edges-transparency 0
	set $this-scalars-transparency 0
	set $this-use-transparency 0
	set $this-interactive_mode "Interactive"
	set $this-bidirectional 0
	set $this-nodes-usedefcolor 0
	set $this-edges-usedefcolor 0
	set $this-faces-usedefcolor 0
	set $this-scalars-usedefcolor 0
	set $this-vectors-usedefcolor 0
	set $this-tensors-usedefcolor 0
	set $this-text-use-default-color 1
	set $this-faces-usetexture  0
	set $this-text-color-r 1.0
	set $this-text-color-g 1.0
	set $this-text-color-b 1.0
	set $this-text-backface-cull 0
	set $this-text-fontsize 1
	set $this-text-precision 2
	set $this-text-render_locations 0
	set $this-text-show-data 1
	set $this-text-show-nodes 0
	set $this-text-show-edges 0
	set $this-text-show-faces 0
	set $this-text-show-cells 0
	set $this-field-name ""
	set $this-field-name-override 0
	set $this-field-name-update 1
	set $this-tensors-emphasis 0.825
	set $this-approx-div 1

	trace variable $this-node_scaleNV w "$this new_node_scale"
	trace variable $this-edge_scaleNV w "$this new_edge_scale"
	trace variable $this-vectors_scaleNV w "$this new_vectors_scale"
	trace variable $this-tensors_scaleNV w "$this new_tensors_scale"
	trace variable $this-scalars_scaleNV w "$this new_scalars_scale"

	trace variable $this-active_tab w "$this switch_to_active_tab"
	trace variable $this-has_vector_data w "$this vector_tab_changed"
	trace variable $this-has_tensor_data w "$this tensor_tab_changed"
	trace variable $this-has_scalar_data w "$this scalar_tab_changed"
	trace variable $this-nodes-as-disks w "$this disk_render_status_changed"
	# no C side component for these variables
	global $this-ns_slider
	set $this-ns_slider "not.set.yet"
	global $this-es_slider
	set $this-es_slider "not.set.yet"
	global $this-vs_slider
	set $this-vs_slider "not.set.yet"
	global $this-ts_slider
	set $this-ts_slider "not.set.yet"
	global $this-ss_slider
	set $this-ss_slider "not.set.yet"
    }

    method new_node_scale {a1 a2 a3} {
	global $this-ns_slider
	set val [set $this-node_scaleNV]
	upvar $this-ns_slider ns_slider
	if {[info exists $this-ns_slider] && [winfo exists $ns_slider]} {
	    $ns_slider newvalue $val
	} else {
	    set $this-node_scale [set $this-node_scaleNV]
	}
	set $this-node_scaleNV -0.0
    }
    
    
    method new_edge_scale {a1 a2 a3} {
	global $this-es_slider
	set val [set $this-edge_scaleNV]
	upvar $this-es_slider es_slider
	if {[info exists $this-es_slider] && [winfo exists $es_slider]} {
	    $es_slider newvalue $val
	} else {
	    set $this-edge_scale [set $this-edge_scaleNV]
	}
	set $this-edge_scaleNV -0.0
    }

    method new_vectors_scale {a1 a2 a3} {
	global $this-vs_slider
	set val [set $this-vectors_scaleNV]
	upvar $this-vs_slider vs_slider
	if {[info exists $this-vs_slider] && [winfo exists $vs_slider]} {
	    $vs_slider newvalue $val
	} else {
	    set $this-vectors_scale [set $this-vectors_scaleNV]
	}
	set $this-vectors_scaleNV -0.0
    }

    method new_tensors_scale {a1 a2 a3} {
	global $this-ts_slider
	set val [set $this-tensors_scaleNV]
	upvar $this-ts_slider ts_slider
	if {[info exists $this-ts_slider] && [winfo exists $ts_slider]} {
	    $ts_slider newvalue $val
	} else {
	    set $this-tensors_scale [set $this-tensors_scaleNV]
	}
	set $this-tensors_scaleNV -0.0
    }
    
    method new_scalars_scale {a1 a2 a3} {
	global $this-ss_slider
	set val [set $this-scalars_scaleNV]
	upvar $this-ss_slider ss_slider
	if {[info exists $this-ss_slider] && [winfo exists $ss_slider]} {
	    $ss_slider newvalue $val
	} else {
	    set $this-scalars_scale [set $this-scalars_scaleNV]
	}
	set $this-scalars_scaleNV -0.0
    }

    method raiseColor {col color colMsg} {
	 global $color
	 set window .ui[modname]
	 if {[winfo exists $window.color]} {
	     SciRaise $window.color
	     return
	 } else {
	     # makeColorPicker now creates the $window.color toplevel.
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

    method addColorSelection {frame text color colMsg} {
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
		 -text $text -command $cmmd
	 
	 #pack the node color frame
	 pack $frame.colorFrame.set_color $frame.colorFrame.col -side left -padx 2
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
    method add_nodes_tab {dof inserting} {
	
	if {$inserting} {
	    set node [$dof.tabs insert 0 -label "Nodes" \
			  -command "$this set_active_tab \"Nodes\""]
	} else {
	    set node [$dof.tabs add -label "Nodes" \
			  -command "$this set_active_tab \"Nodes\""]
	}
	
	checkbutton $node.show_nodes \
		-text "Show Nodes" \
		-command "$this-c toggle_display_nodes" \
		-variable $this-nodes-on

	checkbutton $node.nodes_transparency \
		-text "Enable Transparency (Points Only)" \
		-command "$this-c rerender_nodes" \
		-variable $this-nodes-transparency
	checkbutton $node.udc \
		-text "Use Default Color" \
	        -command "$this-c rerender_nodes" \
		-variable $this-nodes-usedefcolor

	global $this-node_display_type
	
	if {[set $this-nodes-as-disks] == 1} {
	    make_labeled_radio $node.radio \
		    "Node Display Type" "$this-c node_display_type" top \
		    $this-node_display_type \
		{ {Points Points} {Spheres Spheres} \
		      {Boxes Boxes} {Axes Axes} {Disks Disks}}
	} else {
	    make_labeled_radio $node.radio \
		    "Node Display Type" "$this-c node_display_type" top \
		    $this-node_display_type \
		{ {Points Points} {Spheres Spheres}\
		      {Boxes Boxes} {Axes Axes}}
	}

	pack $node.show_nodes $node.nodes_transparency $node.udc $node.radio \
	    -fill y -anchor w

	global $this-ns_slider
	expscale $node.slide -label NodeScale \
	    -orient horizontal \
	    -variable $this-node_scale

	set $this-ns_slider $node.slide

	bind $node.slide.scale <ButtonRelease> "$this-c node_scale"

	iwidgets::labeledframe $node.resolution \
	    -labelpos nw -labeltext "Sphere, Box and Disk Resolution"
	pack $node.resolution -side top -fill x -expand 1

	set res [$node.resolution childsite]
	scale $res.scale -orient horizontal -variable $this-node-resolution \
	    -from 3 -to 20 -showvalue true -resolution 1
	bind $res.scale <ButtonRelease> "$this-c node_resolution_scale"
	pack $res.scale -side top -fill both -expand 1
    }

    # Edges Tab
    method add_edges_tab {dof} {

	set edge [$dof.tabs add -label "Edges" \
		-command "$this set_active_tab \"Edges\""]
	checkbutton $edge.show_edges \
		-text "Show Edges" \
		-command "$this-c toggle_display_edges" \
		-variable $this-edges-on
	checkbutton $edge.edges_transparency \
		-text "Enable Transparency (Lines Only)" \
		-command "$this-c rerender_edges" \
		-variable $this-edges-transparency
	checkbutton $edge.udc \
		-text "Use Default Color" \
	        -command "$this-c rerender_edges" \
		-variable $this-edges-usedefcolor

	make_labeled_radio $edge.radio \
		"Edge Display Type" "$this-c edge_display_type" top \
		$this-edge_display_type {{Cylinders Cylinders} {Lines Lines}}

	pack $edge.show_edges $edge.edges_transparency $edge.udc $edge.radio \
		-side top -fill y -anchor w
	global $this-es_slider
	expscale $edge.slide -label CylinderScale \
		-orient horizontal \
		-variable $this-edge_scale
	set $this-es_slider $edge.slide

	bind $edge.slide.scale <ButtonRelease> "$this-c edge_scale"

	iwidgets::labeledframe $edge.resolution \
	    -labelpos nw -labeltext "Cylinder Resolution"
	pack $edge.resolution -side top -fill x -expand 1

	set res [$edge.resolution childsite]
	scale $res.scale -orient horizontal -variable $this-edge-resolution \
	    -from 3 -to 20 -showvalue true -resolution 1
	bind $res.scale <ButtonRelease> "$this-c edge_resolution_scale"
	pack $res.scale -side top -fill both -expand 1
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
	checkbutton $face.use_transparency \
		-text "Enable Transparency" \
		-command "$this-c rerender_faces" \
		-variable $this-use-transparency
	checkbutton $face.udc \
		-text "Use Default Color" \
	        -command "$this-c rerender_faces" \
		-variable $this-faces-usedefcolor
	checkbutton $face.use_texture \
	        -text "Use 2D texture if available" \
	        -command "$this-c rerender_faces" \
	        -variable $this-faces-usetexture
	pack $face.show_faces $face.use_transparency \
	    $face.udc $face.use_normals $face.use_texture \
	    -side top -fill y -anchor w
    }


    # Vector Tab
    method add_vector_tab {dof} {

	set vector [$dof.tabs add -label "Vectors" \
		-command "$this set_active_tab \"Vectors\""]
	checkbutton $vector.show_vectors \
		-text "Show Vectors" \
		-command "$this-c toggle_display_vectors" \
		-variable $this-vectors-on

	make_labeled_radio $vector.radio \
	    "Vector Display Type" "$this-c data_display_type" top \
	    $this-data_display_type \
	    {{Lines Lines} {Needles Needles} {Cones Cones} \
		 {Arrows Arrows} {Disks Disks}}
	
	checkbutton $vector.normalize_vectors \
		-text "Normalize before scaling" \
		-command "$this-c toggle_normalize" \
		-variable $this-normalize-vectors

	checkbutton $vector.bidirectional \
		-text "Render bidirectionally" \
		-command "$this-c toggle_bidirectional" \
		-variable $this-bidirectional

	checkbutton $vector.usedefcol \
		-text "Use default color" \
		-command "$this-c toggle_bidirectional" \
		-variable $this-vectors-usedefcolor

	pack $vector.show_vectors $vector.radio $vector.normalize_vectors \
	        $vector.bidirectional $vector.usedefcol \
		-side top -fill y -anchor w

	global $this-vs_slider
	expscale $vector.slide -label "Vector Scale" \
		-orient horizontal \
		-variable $this-vectors_scale
	set $this-vs_slider $vector.slide
	bind $vector.slide.scale <ButtonRelease> "$this-c data_scale"


	iwidgets::labeledframe $vector.resolution \
	    -labelpos nw -labeltext "Disk/Cone Resolution"
	pack $vector.resolution -side top -fill x -expand 1

	set res [$vector.resolution childsite]
	scale $res.scale -orient horizontal -variable $this-data-resolution \
	    -from 3 -to 20 -showvalue true -resolution 1
	bind $res.scale <ButtonRelease> "$this-c data_resolution_scale"
	pack $res.scale -side top -fill both -expand 1
    }


    # Tensor Tab
    method add_tensor_tab {dof} {

	set tensor [$dof.tabs add -label "Tensors" \
		-command "$this set_active_tab \"Tensors\""]
	checkbutton $tensor.show_tensors \
		-text "Show Tensors" \
		-command "$this-c toggle_display_tensors" \
		-variable $this-tensors-on

	make_labeled_radio $tensor.radio \
	    "Tensor Display Type" "$this-c data_display_type" top \
	    $this-tensor_display_type \
	    {{Boxes Boxes} {Ellipsoids Ellipsoids} \
		 {Superquadrics Superquadrics} \
		 {"Colored Boxes" "Colored Boxes"}}
	
	checkbutton $tensor.usedefcol \
		-text "Use default color" \
		-command "$this-c data_display_type" \
		-variable $this-tensors-usedefcolor

	pack $tensor.show_tensors $tensor.radio $tensor.usedefcol \
		-side top -fill y -anchor w
	
	global $this-ts_slider
	expscale $tensor.slide -label "Tensor Scale" \
		-orient horizontal \
		-variable $this-tensors_scale
	set $this-ts_slider $tensor.slide
	bind $tensor.slide.scale <ButtonRelease> "$this-c data_scale"

	iwidgets::labeledframe $tensor.resolution \
	    -labelpos nw -labeltext "Ellipse Resolution"
	pack $tensor.resolution -side top -fill x -expand 1

	set res [$tensor.resolution childsite]
	scale $res.scale -orient horizontal -variable $this-data-resolution \
	    -from 3 -to 20 -showvalue true -resolution 1
	bind $res.scale <ButtonRelease> "$this-c data_resolution_scale"
	pack $res.scale -side top -fill both -expand 1
    }

    # Scalar Tab
    method add_scalar_tab {dof} {

	set scalar [$dof.tabs add -label "Scalars" \
		-command "$this set_active_tab \"Scalars\""]
	checkbutton $scalar.show_scalars \
		-text "Show Scalars" \
		-command "$this-c toggle_display_scalars" \
		-variable $this-scalars-on

	checkbutton $scalar.transparency \
		-text "Enable Transparency (Points Only)" \
		-command "$this-c data_scale" \
		-variable $this-scalars-transparency

	checkbutton $scalar.usedefcol \
		-text "Use Default Color" \
		-command "$this-c data_scale" \
		-variable $this-scalars-usedefcolor

	make_labeled_radio $scalar.radio \
	    "Scalar Display Type" "$this-c data_display_type" top \
	    $this-scalar_display_type \
	    {{Points Points} {Spheres Spheres} \
		 {"Scaled Spheres" "Scaled Spheres"}}
	
	pack $scalar.show_scalars $scalar.transparency \
	    $scalar.usedefcol $scalar.radio \
	    -side top -fill y -anchor w

	global $this-ss_slider
	expscale $scalar.slide -label "Scalar Scale" \
		-orient horizontal \
		-variable $this-scalars_scale
	set $this-ss_slider $scalar.slide
	bind $scalar.slide.scale <ButtonRelease> "$this-c data_scale"

	iwidgets::labeledframe $scalar.resolution \
	    -labelpos nw -labeltext "Sphere Resolution"
	pack $scalar.resolution -side top -fill x -expand 1

	set res [$scalar.resolution childsite]
	scale $res.scale -orient horizontal -variable $this-data-resolution \
	    -from 3 -to 20 -showvalue true -resolution 1
	bind $res.scale <ButtonRelease> "$this-c data_resolution_scale"
	pack $res.scale -side top -fill both -expand 1
    }

    # Text Tab
    method add_text_tab {dof} {
	set text [$dof.tabs add -label "Text" \
		-command "$this set_active_tab \"Text\""]
	checkbutton $text.show_text \
		-text "Show Text" \
		-command "$this-c toggle_display_text" \
		-variable $this-text-on

	frame $text.def_col -borderwidth 2

	checkbutton $text.backfacecull \
	    -text "Cull backfacing text if possible" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-backface-cull

	checkbutton $text.locations \
	    -text "Render indices as locations" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-render_locations

	frame $text.show 
	checkbutton $text.show.data \
	    -text "Show data values" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-show-data
	checkbutton $text.show.nodes \
	    -text "Show node indices" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-show-nodes
	checkbutton $text.show.edges \
	    -text "Show edge indices" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-show-edges
	checkbutton $text.show.faces \
	    -text "Show face indices" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-show-faces
	checkbutton $text.show.cells \
	    -text "Show cell indices" \
	    -command "$this-c rerender_text" \
	    -variable $this-text-show-cells

	make_labeled_radio $text.size \
	    "Text Size:" "$this-c rerender_text" left \
	    $this-text-fontsize \
	    {{"XS" 0} {"S" 1} {"M" 2} {"L" 3} {"XL" 4}}

	pack $text.show.data $text.show.nodes $text.show.edges \
	    $text.show.faces $text.show.cells \
	    -side top -fill y -anchor w
	
	checkbutton $text.use_def_col \
		-text "Use text color" \
		-command "$this-c rerender_text" \
		-variable $this-text-use-default-color

	addColorSelection $text.def_col "Text Color" $this-text-color \
	    "text_color_change"

	frame $text.precision
	label $text.precision.label -text "Text Precision  "
	scale $text.precision.scale -orient horizontal \
	    -variable $this-text-precision -from 1 -to 16 \
	    -showvalue true -resolution 1
	bind $text.precision.scale <ButtonRelease> "$this-c rerender_text"
	pack $text.precision.label -side left -anchor s -fill x
	pack $text.precision.scale -side left -anchor n -fill x
	
	pack $text.show_text $text.show $text.backfacecull $text.locations \
	    $text.use_def_col $text.def_col $text.size \
	    $text.precision -side top -fill y -anchor w
    }

    method disk_render_status_changed {name1 name2 op} {
	#puts stdout "called disk_render_status_changed"
	set window .ui[modname]
	if {[winfo exists $window]} {
	    set dof [$window.options.disp.frame_title childsite]
	    $dof.tabs delete "Nodes"
	    add_nodes_tab $dof 1
	    $dof.tabs view [set $this-active_tab]
	}
    }

    method vector_tab_changed {name1 name2 op} {
	global $this-has_vector_data

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

    method tensor_tab_changed {name1 name2 op} {
	global $this-has_tensor_data

	set window .ui[modname]
	if {[winfo exists $window]} {
	    set dof [$window.options.disp.frame_title childsite]	
	    if {[set $name1] == 1} { 
		add_tensor_tab $dof
		$dof.tabs view [set $this-active_tab]
	    } else {
		$dof.tabs delete "Tensors"
	    }
	}
    }

    method scalar_tab_changed {name1 name2 op} {
	global $this-has_scalar_data

	set window .ui[modname]
	if {[winfo exists $window]} {
	    set dof [$window.options.disp.frame_title childsite]	
	    if {[set $name1] == 1} { 
		add_scalar_tab $dof
		$dof.tabs view [set $this-active_tab]
	    } else {
		$dof.tabs delete "Scalars"
	    }
	}
    }

    method ui {} {
	set window .ui[modname]
	if {[winfo exists $window]} {
	    return
	}
	toplevel $window
	#wm minsize $window 380 548

	#frame for all options to live
	frame $window.options
 
	# node frame holds ui related to vert display (left side)
	frame $window.options.disp -borderwidth 2
	pack $window.options.disp -padx 2 -pady 2 -side left \
		-fill both -expand 1

	# Display Options
	iwidgets::labeledframe $window.options.disp.frame_title \
		-labelpos nw -labeltext "Display Options"
	set dof [$window.options.disp.frame_title childsite]

	iwidgets::tabnotebook  $dof.tabs -height 380 -width 330 \
	    -raiseselect true 
	#label $window.options.disp.frame_title -text "Display Options"

	add_nodes_tab $dof 0
	add_edges_tab $dof
	add_faces_tab $dof
	add_text_tab $dof
	if {[set $this-has_vector_data] == 1} {
	    add_vector_tab $dof
	}
	if {[set $this-has_tensor_data] == 1} {
	    add_tensor_tab $dof
	}
	if {[set $this-has_scalar_data] == 1} {
	    add_scalar_tab $dof
	}

	global $this-active_tab
	global $this-interactive_mode
	# view the active tab
	if [catch "$dof.tabs view [set $this-active_tab]"] {
	    catch "$dof.tabs view 0"
	}
	$dof.tabs configure -tabpos "n"

	pack $dof.tabs -side top -fill x -expand yes -padx 2 -pady 2

	#pack notebook frame
	pack $window.options.disp.frame_title -side top -expand yes -fill x
	
	#add bottom frame for execute and dismiss buttons
	frame $window.control -relief groove -borderwidth 2 -width 500
	frame $window.def
	frame $window.def.vals
	frame $window.def.col
	frame $window.def.col.f
	frame $window.def.col.le

	pack $window.def.col $window.def.vals -side left -padx 10
	label $window.def.col.le.approxl -text "PWL Approx Div:"
	entry $window.def.col.le.approx -textvar $this-approx-div -width 4

	bind $window.def.col.le.approx <KeyRelease> "$this-c approx"



	addColorSelection $window.def.col.f "Default Color" \
	    $this-def-color "default_color_change"

	button $window.def.vals.calcdefs -text "Calculate Defaults" \
		-command "$this-c calcdefs"
	checkbutton $window.def.vals.use_defaults \
		-text "Use Defaults" \
		-variable $this-use-defaults

	pack $window.def.col.f $window.def.col.le -side top -pady 2 -anchor w
	pack $window.def.col.le.approxl $window.def.col.le.approx -side left
	pack $window.def.vals.use_defaults $window.def.vals.calcdefs \
	    -side top -pady 2

	frame $window.fname -borderwidth 2
	label $window.fname.label -text "Field Name"
	entry $window.fname.entry -textvar $this-field-name
	checkbutton $window.fname.override \
		-text "Override" \
		-variable $this-field-name-override

	TooltipMultiWidget "$window.fname.entry $window.fname.label" \
	    "Enter (optional) Field Name here.  The name will be displayed\nin the Viewer Window's list of Objects."
	
	pack $window.fname.label $window.fname.entry -side left
	pack $window.fname.override -side left -padx 6
	pack $window.fname -anchor w -padx 6 -pady 6


	## Cylinder and Sphere Resolution
	#iwidgets::labeledframe $window.resolution \
	#	-labelpos nw -labeltext "Cylinder and Sphere Resolution"
	#set res [$window.resolution childsite]
	#
	#scale $res.scale -orient horizontal -variable $this-resolution \
	#	-from 3 -to 20 -showvalue true -resolution 1
	#
	#bind $res.scale <ButtonRelease> "$this-c resolution_scale"

	# execute policy
	make_labeled_radio $window.control.exc_policy \
		"Execute Policy" "$this-c execute_policy" top \
		$this-interactive_mode \
		{{"Interactively update" Interactive} \
		{"Execute button only" OnExecute}}

	#pack $res.scale -side top -fill both -expand 1

	pack $window.options -padx 2 -pady 2 -side top -fill x -expand 1
	#pack $window.resolution -padx 2 -pady 2 -side top -fill x -expand 1
	pack $window.def $window.control \
	    -padx 2 -pady 2 -side top

	pack $window.control.exc_policy -side top -fill both

	frame $window.control.excdis -borderwidth 2
	pack $window.control.excdis -padx 4 -pady 4 -side top -fill both

	makeSciButtonPanel $window.control.excdis $window $this
	moveToCursor $window

	pack $window.control -padx 4 -pady 4 -side top -fill both
    }
}


















