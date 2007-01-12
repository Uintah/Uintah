itcl_class Teem_Gage_GageProbe {
    inherit Module
    constructor {config} {
        set name GageProbe
        set_defaults
    }
    
    method set_defaults {} {
	    
	    global $this-field_kind_
	    global $this-field_kind_list
	    global $this-otype_
	    global $this-quantity_
	    global $this-valuesType_
	    global $this-valuesNumParm1_
	    global $this-valuesNumParm2_
	    global $this-valuesNumParm3_
	    global $this-dType_
	    global $this-dNumParm1_
	    global $this-dNumParm2_
	    global $this-dNumParm3_
	    global $this-ddType_
	    global $this-ddNumParm1_
	    global $this-ddNumParm2_
	    global $this-ddNumParm3_
	    global $this-quantityListVec_
	    global $this-quantityListScl_
	    global $this-quantityDescListVec_
	    global $this-quantityDescListScl_
	    
	    set $this-field_kind_ "Scalar"
	    set $this-field_kind_list ""
	    set $this-otype_ "double"
	    set $this-quantity_ "value"
	    set $this-valuesType_ "cubic"
	    set $this-valuesNumParm1_ 0.0
	    set $this-valuesNumParm2_ 0.5
	    
	    set $this-dType_ "cubicd"
	    set $this-dNumParm1_ 1.0
	    set $this-dNumParm2_ 0.0
	    
	    set $this-ddType_ "cubicdd"
	    set $this-ddNumParm1_ 1.0
	    set $this-ddNumParm2_ 0.0
	    
	    set $this-quantityListVec_ {"item 1" "item 2" "item 3"}
	    set $this-quantityListScl_ "dfg"
	    set $this-quantityDescListVec_ "sdfgh"
	    set $this-quantityDescListScl_ "sdfghedd"
	    
    }
    
    method changeDataset { w } {
	    global $w
	    global $this-field_kind_
	    global $this-quantity_
	    
	    set $this-field_kind_ [$w get]
	    if {[set $this-field_kind_] == "Scalar" } {
	      $this change_quantity_menu_to_scalar
	    } elseif {[set $this-field_kind_] == "Vector" } {
	      $this change_quantity_menu_to_vector
	    }
	    $this update_quantity_menu
    }
    
    method update_values_type { w d1 d2 d3 e1 e2 e3 } {
	    global $w
	    global $d1
	    global $d2
	    global $d3
	    global $e1
	    global $e2
	    global $e3
	    global $this-valuesType_
	    
	    set $this-valuesType_ [$w get]
	    $this update_num_parms $w $d1 $d2 $d3 $e1 $e2 $e3
	    
    }
    
    method update_d_type { w d1 d2 d3 e1 e2 e3 } {
	    global $w
	    global $d1
	    global $d2
	    global $d3
	    global $e1
	    global $e2
	    global $e3
	    global $this-dType_
	    
	    set $this-dType_ [$w get]
	    $this update_num_parms $w $d1 $d2 $d3 $e1 $e2 $e3
	    
    }
    
    method update_dd_type { w d1 d2 d3 e1 e2 e3 } {
	    global $w
	    global $d1
	    global $d2
	    global $d3
	    global $e1
	    global $e2
	    global $e3
	    global $this-ddType_
	    
	    set $this-ddType_ [$w get]
	    $this update_num_parms $w $d1 $d2 $d3 $e1 $e2 $e3
	    
    }
    
    method update_otype { w } {
	    global $w
	    global $this-otype_
	    
	    set $this-otype_ [$w get]
    }
    
    method update_num_parms { w d1 d2 d3 e1 e2 e3 } {
	    global $w
	    global $d1
	    global $d2
	    global $d3
	    global $e1
	    global $e2
	    global $e3
	    
	    if {([$w get] == "gaussian") || ([$w get] == "gaussiand") \
	    	|| ([$w get] == "gaussiandd")} {
		#change the first description to sigma
		$d1 configure -text "sigma"
	    } else {
		#change the first description to scale
		$d1 configure -text "scale"
	    }
	    
	    if { [$w get] == "cubic" } {
		#set the second description to B
		$d2 configure -text "B"
		#enable the second entry
		$e2 configure -state normal
	    } elseif { [$w get] == "quartic" } {
		#set the second description to A
		$d2 configure -text "A"
		#enable the second entry
		$e2 configure -state normal
	    } elseif { ([$w get] == "hann") || ([$w get] == "hannd") \
		|| ([$w get] == "hanndd") || ([$w get] == "blackman") \
	    	|| ([$w get] == "blackmand") || ([$w get] == "blackmandd") \
	    	|| ([$w get] == "gaussian") || ([$w get] == "gaussiand") \
	    	|| ([$w get] == "gaussiandd")} {
		#set the second description to cut-off
		$d2 configure -text "cut-off"
		#enable the second entry
		$e2 configure -state normal
	    } else {
		#set the second label to empty string
		$d2 configure -text " "
		#disable second entry
		$e2 configure -state disabled
	    }
	    
	    if {([$w get] == "cubic") || ([$w get] == "cubicd") \
	    	|| ([$w get] == "cubicdd")} {
		#set third description to C
		$d3 configure -text "C"
		#enable the third entry
		$e3 configure -state normal
	    } else {
		#set third description to empty string
		$d3 configure -text " "
		#disable the third entry
		$e3 configure -state disabled
	    }
    }
    
    method update_quantity_description { w } {
	global $w
	global $this-quantity_
	set $this-quantity_ [$w get]
	if {[set $this-quantity_] == "value" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "reconstructed scalar data value"
	} elseif {[set $this-quantity_] == "gradient vector" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient vector, un-normalized"
	} elseif {[set $this-quantity_] == "gradient magnitude" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient magnitude \n (length of  \
		gradient vector)" -justify left
	} elseif {[set $this-quantity_] == "normalized gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "projection into tangent \n (perp \
		space of normal)" -justify left
	} elseif {[set $this-quantity_] == "tangent projector" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "normalized gradient vector"
	} elseif {[set $this-quantity_] == "Hessian" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "3x3 Hessian matrix"
	} elseif {[set $this-quantity_] == "Laplacian" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "Laplacian"
	} elseif {[set $this-quantity_] == "Frob(Hessian)" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "Frobenius norm of Hessian"
	} elseif {[set $this-quantity_] == "2nd DD along gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "2nd directional derivative \n along \
		gradient" -justify left
	} elseif {[set $this-quantity_] == "geometry tensor" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "geometry tensor"
	} elseif {[set $this-quantity_] == "kappa1" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "1st principal curvature (K1)"
	} elseif {[set $this-quantity_] == "kappa2" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "2nd principal curvature (K2)"
	} elseif {[set $this-quantity_] == "total curvature" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "total curvature (L2 norm of K1, K2)"
	} elseif {[set $this-quantity_] == "shape trace" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "shape trace = \n (K1+K2)/(total \
		curvature)" -justify left
	} elseif {[set $this-quantity_] == "shape index" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "Koenderink's shape index"
	} elseif {[set $this-quantity_] == "mean curvature" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "mean curvature = (K1+K2)/2"
	} elseif {[set $this-quantity_] == "Gaussian curvature" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gaussian curvature = K1*K2"
	} elseif {[set $this-quantity_] == "1st curvature direction" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "1st principal curvature direction"
	} elseif {[set $this-quantity_] == "2nd curvature direction" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "2nd principal curvature direction"
	} elseif {[set $this-quantity_] == "flowline curvature" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "curvature of normal streamline"
	} elseif {[set $this-quantity_] == "Hessian eigenvalues" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "Hessian's eigenvalues"
	} elseif {[set $this-quantity_] == "median" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "median of iv3 cache \n (not weighted by \
		any filter (yet))" -justify left
	
	} elseif {[set $this-quantity_] == "vector" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "component-wise-interpolated \n \
		vector" -justify left
	} elseif {[set $this-quantity_] == "vector0" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "vector\[0\]" -justify left
	} elseif {[set $this-quantity_] == "vector1" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "vector\[1\]" -justify left
	} elseif {[set $this-quantity_] == "vector2" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "vector\[2\]" -justify left
	} elseif {[set $this-quantity_] == "length" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "length of vector" -justify left
	} elseif {[set $this-quantity_] == "normalized" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "normalized vector" -justify left
	} elseif {[set $this-quantity_] == "Jacobian" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "3x3 Jacobian" -justify left
	} elseif {[set $this-quantity_] == "divergence" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "divergence" -justify left
	} elseif {[set $this-quantity_] == "curl" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "curl" -justify left
	} elseif {[set $this-quantity_] == "curl norm" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "curl magnitude" -justify left
	} elseif {[set $this-quantity_] == "helicity" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "helicity: dot(vector,curl)" -justify left
	} elseif {[set $this-quantity_] == "normalized helicity" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "normalized helicity" -justify left
	} elseif {[set $this-quantity_] == "lambda2" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "lambda2 value for \n vortex \
		characterization" -justify left
	} elseif {[set $this-quantity_] == "vector hessian" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "3x3x3 second-order \n vector \
		derivative" -justify left
	} elseif {[set $this-quantity_] == "div gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient of divergence" -justify left
	} elseif {[set $this-quantity_] == "curl gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "3x3 derivative of curl" -justify left
	} elseif {[set $this-quantity_] == "curl norm gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient of curl norm" -justify left
	} elseif {[set $this-quantity_] == "normalized curl norm gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "normalized gradient of \n curl \
		norm" -justify left
	} elseif {[set $this-quantity_] == "helicity gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient of helicity" -justify left
	} elseif {[set $this-quantity_] == "directional helicity derivative" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "directional derivative \n of helicity \
		along flow" -justify left
	} elseif {[set $this-quantity_] == "projected helicity gradient" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "projection of the helicity \n gradient \
		onto plane \n orthogonal to flow" -justify left
	} elseif {[set $this-quantity_] == "gradient0" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient of 1st \n component of \
		vector" -justify left
	} elseif {[set $this-quantity_] == "gradient1" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient of 2nd \n component of \
		vector" -justify left
	} elseif {[set $this-quantity_] == "gradient2" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "gradient of 3rd \n component of \
		vector" -justify left
	} elseif {[set $this-quantity_] == "multigrad" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "multi-gradient: \n sum of outer \
		products of gradients" -justify left
	} elseif {[set $this-quantity_] == "frob(multigrad)" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "frob norm of multi-gradient" -justify left
	} elseif {[set $this-quantity_] == "multigrad eigenvalues" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "eigenvalues of multi-gradient" -justify left
	} elseif {[set $this-quantity_] == "multigrad eigenvectors" } {
		.ui[modname].f.quantityFrame.description \
		configure -text "eigenvectors of multi-gradient" -justify left
	}
    }

    method set_list { list } {
	global $list
        set w .ui[modname]

	global $this-field_kind_list
	set $this-field_kind_list $list
	puts "got this: [set $this-field_kind_list]"
    }
    
    #update the variable associated with the value of the measure(quantity) menu
    method update_quantity_menu {} {
	global $this-quantity_
	.ui[modname].f.quantityFrame.quantityMenu select [set $this-quantity_]
    }
    
    method change_quantity_menu_to_scalar {} {
	.ui[modname].f.quantityFrame.quantityMenu delete 0 27
	
	.ui[modname].f.quantityFrame.quantityMenu insert end "value" \
	"gradient vector" \
	"gradient magnitude" \
	"normalized gradient" \
	"tangent projector" \
	"Hessian" \
	"Laplacian" \
	"Frob(Hessian)" \
	"2nd DD along gradient" \
	"geometry tensor"\
	"kappa1" \
	"kappa2" \
	"total curvature" \
	"shape trace" \
	"shape index"\
	"mean curvature" \
	"Gaussian curvature" \
	"1st curvature direction" \
	"2nd curvature direction" \
	"flowline curvature" \
	"Hessian eigenvalues" \
	"median"
	
	global $this-quantity_
	set $this-quantity_ "value"
    }
    
    method change_quantity_menu_to_vector {} {
	.ui[modname].f.quantityFrame.quantityMenu delete 0 21
	
	.ui[modname].f.quantityFrame.quantityMenu insert end "vector" \
	"vector0" \
	"vector1" \
	"vector2" \
	"length" \
	"normalized" \
	"Jacobian" \
	"divergence" \
	"curl" \
	"curl norm" \
	"helicity" \
	"normalized helicity" \
	"lambda2" \
	"vector hessian" \
	"div gradient" \
	"curl gradient" \
	"curl norm gradient" \
	"normalized curl norm gradient" \
	"helicity gradient" \
        "directional helicity derivative" \
	"projected helicity gradient" \
	"gradient0" \
	"gradient1" \
	"gradient2" \
	"multigrad" \
	"frob(multigrad)"  \
	"multigrad eigenvalues" \
	"multigrad eigenvectors"
	
	global $this-quantity_
	set $this-quantity_ "vector"
    }
    
    
    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }
        toplevel $w
	
	puts "here: w is $w, this is $this"
	
        frame $w.f
	pack $w.f -padx 2 -pady 2 -side top -expand yes
	
	global $this-field_kind_
	global $this-field_kind_list
	global $this-otype_
	
	#frame describing the field kind
	frame $w.f.fieldKindFrame
	
	iwidgets::optionmenu $w.f.fieldKindFrame.fieldKindMenu \
	-labeltext "Kind of Input Field: " -labelpos w \
	-command "$this changeDataset $w.f.fieldKindFrame.fieldKindMenu"
	
	eval $w.f.fieldKindFrame.fieldKindMenu insert end "Scalar Vector"
	$w.f.fieldKindFrame.fieldKindMenu select [set $this-field_kind_]
	
	grid $w.f.fieldKindFrame.fieldKindMenu -column 0 -row 0 -sticky w
	
	#frame describing the quantity the user wishes to measure
	frame $w.f.quantityFrame -relief ridge
	label $w.f.quantityFrame.quantityLabel -text "Quantity to Measure: "
	iwidgets::optionmenu $w.f.quantityFrame.quantityMenu \
	-command "$this update_quantity_description \
	$w.f.quantityFrame.quantityMenu"
	$w.f.quantityFrame.quantityMenu insert end "value" \
	"gradient vector" \
	"gradient magnitude" \
	"normalized gradient" \
	"tangent projector" \
	"Hessian" \
	"Laplacian" \
	"Frob(Hessian)" \
	"2nd DD along gradient" \
	"geometry tensor"\
	"kappa1" \
	"kappa2" \
	"total curvature" \
	"shape trace" \
	"shape index"\
	"mean curvature" \
	"Gaussian curvature" \
	"1st curvature direction" \
	"2nd curvature direction" \
	"flowline curvature" \
	"Hessian eigenvalues" \
	"median"
	$w.f.quantityFrame.quantityMenu select [set $this-quantity_]
	
	label $w.f.quantityFrame.descriptionLabel -text "Description: "
	label $w.f.quantityFrame.description \
	    -text "reconstructed scalar data value"
	label $w.f.quantityFrame.otypeLabel -text "Output Type: "
	iwidgets::optionmenu $w.f.quantityFrame.otypeMenu -command "$this \
	update_otype $w.f.quantityFrame.otypeMenu"
	$w.f.quantityFrame.otypeMenu insert end	"double" "float" "default"
	$w.f.quantityFrame.otypeMenu select [set $this-otype_]
	
	grid $w.f.quantityFrame.quantityLabel -column 0 -row 0 -sticky w
	grid configure $w.f.quantityFrame.quantityMenu -column 1 -row \
	0 -sticky w
	grid configure $w.f.quantityFrame.descriptionLabel -column 0 \
	-row 1 -sticky w
	grid configure $w.f.quantityFrame.description -column 1 -row 1 -sticky w
	grid configure $w.f.quantityFrame.otypeLabel -column 0 -row 2 -sticky w
	grid configure $w.f.quantityFrame.otypeMenu -column 1 -row 2 -sticky w
	
        $w.f.quantityFrame configure -borderwidth 2
	
	#frame for specifying kernels
	frame $w.f.kernelsFrame -borderwidth 3 -relief sunken
	
	label $w.f.kernelsFrame.kernelsLabel \
	-text "Kernels to Use: " -padx 5 -pady 10
	
	##subframe for specifying the kernel parameters for 
	##reconstructing values
	frame $w.f.kernelsFrame.valuesFrame -borderwidth 3 -relief raised
	
	label $w.f.kernelsFrame.valuesFrame.valuesLabel \
	-text "Reconstructing Values" -padx 5 -pady 5
	label $w.f.kernelsFrame.valuesFrame.valuesTypeLabel \
	-text "Kernel Type: " -padx 10
	iwidgets::optionmenu $w.f.kernelsFrame.valuesFrame.valuesTypeMenu \
	-command "$this update_values_type \
	$w.f.kernelsFrame.valuesFrame.valuesTypeMenu \
	$w.f.kernelsFrame.valuesFrame.valuesNumParm1Desc \
	$w.f.kernelsFrame.valuesFrame.valuesNumParm2Desc \
	$w.f.kernelsFrame.valuesFrame.valuesNumParm3Desc \
	$w.f.kernelsFrame.valuesFrame.valuesNumParm1Entry \
	$w.f.kernelsFrame.valuesFrame.valuesNumParm2Entry \
	$w.f.kernelsFrame.valuesFrame.valuesNumParm3Entry "
	$w.f.kernelsFrame.valuesFrame.valuesTypeMenu insert end	"zero" \
	"box" \
	"tent" \
	"cubic" \
	"quartic" \
	"gaussian" \
	"hann" \
	"blackman"
	$w.f.kernelsFrame.valuesFrame.valuesTypeMenu \
	select [set $this-valuesType_]
	
	label $w.f.kernelsFrame.valuesFrame.valuesNumParmLabel \
	-text "Numeric Parameters: " -padx 10
	label $w.f.kernelsFrame.valuesFrame.valuesNumParm1Desc \
	-text "scale" -padx 20
	label $w.f.kernelsFrame.valuesFrame.valuesNumParm2Desc \
	-text "B" -padx 20
	label $w.f.kernelsFrame.valuesFrame.valuesNumParm3Desc \
	-text "C" -padx 20
	entry $w.f.kernelsFrame.valuesFrame.valuesNumParm1Entry \
	-width 6 -textvariable $this-valuesNumParm1_
	entry $w.f.kernelsFrame.valuesFrame.valuesNumParm2Entry \
	-width 6 -textvariable $this-valuesNumParm2_
	entry $w.f.kernelsFrame.valuesFrame.valuesNumParm3Entry \
	-width 6 -textvariable $this-valuesNumParm3_
	
	grid $w.f.kernelsFrame.kernelsLabel -row 0 -sticky w
	grid configure $w.f.kernelsFrame.valuesFrame -row 1 -sticky ew
	grid $w.f.kernelsFrame.valuesFrame.valuesLabel -row 0 -column 0 \
	-sticky w
	grid $w.f.kernelsFrame.valuesFrame.valuesTypeLabel -row 1 -column 0 \
	-sticky w
	grid $w.f.kernelsFrame.valuesFrame.valuesTypeMenu -row 1 -column 2
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParmLabel -row 2 -column 0 \
	-sticky w 
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParm1Desc -row 3 -column 0 \
	-sticky w
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParm2Desc -row 4 -column 0 \
	-sticky w
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParm3Desc -row 5 -column 0 \
	-sticky w
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParm1Entry -row 3 -column 2
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParm2Entry -row 4 -column 2
	grid $w.f.kernelsFrame.valuesFrame.valuesNumParm3Entry -row 5 -column 2
	
	##subframe for specifying the kernel parameters for 
	##measuring 1st derivative
	frame $w.f.kernelsFrame.dFrame -borderwidth 3 -relief raised
	
	label $w.f.kernelsFrame.dFrame.dLabel \
	-text "Measuring 1st Derivative" -padx 5 -pady 5
	label $w.f.kernelsFrame.dFrame.dTypeLabel -text "Kernel Type: " -padx 10
	iwidgets::optionmenu $w.f.kernelsFrame.dFrame.dTypeMenu \
	-command "$this update_d_type \
	$w.f.kernelsFrame.dFrame.dTypeMenu \
	$w.f.kernelsFrame.dFrame.dNumParm1Desc \
	$w.f.kernelsFrame.dFrame.dNumParm2Desc \
	$w.f.kernelsFrame.dFrame.dNumParm3Desc \
	$w.f.kernelsFrame.dFrame.dNumParm1Entry \
	$w.f.kernelsFrame.dFrame.dNumParm2Entry \
	$w.f.kernelsFrame.dFrame.dNumParm3Entry"
	$w.f.kernelsFrame.dFrame.dTypeMenu insert end "zero" \
	"box" \
	"forwdiff" \
	"centdiff" \
	"cubicd" \
	"quarticd" \
	"gaussiand" \
	"hannd" \
	"blackmand"
	$w.f.kernelsFrame.dFrame.dTypeMenu select [set $this-dType_]
	
	label $w.f.kernelsFrame.dFrame.dNumParmLabel \
	-text "Numeric Parameters: " -padx 10
	label $w.f.kernelsFrame.dFrame.dNumParm1Desc \
	-text "scale" -padx 20
	label $w.f.kernelsFrame.dFrame.dNumParm2Desc \
	-text "B" -padx 20
	label $w.f.kernelsFrame.dFrame.dNumParm3Desc \
	-text "C" -padx 20
	entry $w.f.kernelsFrame.dFrame.dNumParm1Entry \
	-width 6 -textvariable $this-dNumParm1_ 
	entry $w.f.kernelsFrame.dFrame.dNumParm2Entry \
	-width 6 -textvariable $this-dNumParm2_
	entry $w.f.kernelsFrame.dFrame.dNumParm3Entry \
	-width 6 -textvariable $this-dNumParm3_
	
	grid $w.f.kernelsFrame.kernelsLabel -row 0 -sticky w
	grid configure $w.f.kernelsFrame.dFrame -row 2 -sticky ew
	grid $w.f.kernelsFrame.dFrame.dLabel -row 0 -column 0 -sticky w
	grid $w.f.kernelsFrame.dFrame.dTypeLabel -row 1 -column 0 -sticky w
	grid $w.f.kernelsFrame.dFrame.dTypeMenu -row 1 -column 2
	grid $w.f.kernelsFrame.dFrame.dNumParmLabel -row 2 -column 0 -sticky w 
	grid $w.f.kernelsFrame.dFrame.dNumParm1Desc -row 3 -column 0 -sticky w
	grid $w.f.kernelsFrame.dFrame.dNumParm2Desc -row 4 -column 0 -sticky w
	grid $w.f.kernelsFrame.dFrame.dNumParm3Desc -row 5 -column 0 -sticky w
	grid $w.f.kernelsFrame.dFrame.dNumParm1Entry -row 3 -column 2
	grid $w.f.kernelsFrame.dFrame.dNumParm2Entry -row 4 -column 2
	grid $w.f.kernelsFrame.dFrame.dNumParm3Entry -row 5 -column 2
	
	##subframe for specifying the kernel parameters for 
	##measuring 2nd derivative
	frame $w.f.kernelsFrame.ddFrame -borderwidth 3 -relief raised
	
	label $w.f.kernelsFrame.ddFrame.ddLabel \
	-text "Measuring 2nd Derivative" -padx 5 -pady 5
	label $w.f.kernelsFrame.ddFrame.ddTypeLabel \
	-text "Kernel Type: " -padx 10
	iwidgets::optionmenu $w.f.kernelsFrame.ddFrame.ddTypeMenu \
	-command "$this update_dd_type \
	$w.f.kernelsFrame.ddFrame.ddTypeMenu \
	$w.f.kernelsFrame.ddFrame.ddNumParm1Desc \
	$w.f.kernelsFrame.ddFrame.ddNumParm2Desc \
	$w.f.kernelsFrame.ddFrame.ddNumParm3Desc \
	$w.f.kernelsFrame.ddFrame.ddNumParm1Entry \
	$w.f.kernelsFrame.ddFrame.ddNumParm2Entry \
	$w.f.kernelsFrame.ddFrame.ddNumParm3Entry"
	$w.f.kernelsFrame.ddFrame.ddTypeMenu insert end "zero" \
	"box" \
	"cubicdd" \
	"quarticdd" \
	"gaussiandd" \
	"hanndd" \
	"blackmandd"
	$w.f.kernelsFrame.ddFrame.ddTypeMenu select [set $this-ddType_]
	
	label $w.f.kernelsFrame.ddFrame.ddNumParmLabel \
	-text "Numeric Parameters: " -padx 10
	label $w.f.kernelsFrame.ddFrame.ddNumParm1Desc \
	-text "scale" -padx 20
	label $w.f.kernelsFrame.ddFrame.ddNumParm2Desc \
	-text "B" -padx 20
	label $w.f.kernelsFrame.ddFrame.ddNumParm3Desc \
	-text "C" -padx 20
	entry $w.f.kernelsFrame.ddFrame.ddNumParm1Entry \
	-width 6 -textvariable $this-ddNumParm1_
	entry $w.f.kernelsFrame.ddFrame.ddNumParm2Entry \
	-width 6 -textvariable $this-ddNumParm2_
	entry $w.f.kernelsFrame.ddFrame.ddNumParm3Entry \
	-width 6 -textvariable $this-ddNumParm3_
	
	grid $w.f.kernelsFrame.kernelsLabel -row 0 -sticky w
	grid configure $w.f.kernelsFrame.ddFrame -row 3 -sticky ew
	grid $w.f.kernelsFrame.ddFrame.ddLabel -row 0 -column 0 -sticky w
	grid $w.f.kernelsFrame.ddFrame.ddTypeLabel -row 1 -column 0 -sticky w
	grid $w.f.kernelsFrame.ddFrame.ddTypeMenu -row 1 -column 2
	grid $w.f.kernelsFrame.ddFrame.ddNumParmLabel -row 2 -column 0 -sticky w 
	grid $w.f.kernelsFrame.ddFrame.ddNumParm1Desc -row 3 -column 0 -sticky w
	grid $w.f.kernelsFrame.ddFrame.ddNumParm2Desc -row 4 -column 0 -sticky w
	grid $w.f.kernelsFrame.ddFrame.ddNumParm3Desc -row 5 -column 0 -sticky w
	grid $w.f.kernelsFrame.ddFrame.ddNumParm1Entry -row 3 -column 2
	grid $w.f.kernelsFrame.ddFrame.ddNumParm2Entry -row 4 -column 2
	grid $w.f.kernelsFrame.ddFrame.ddNumParm3Entry -row 5 -column 2
		
	pack $w.f.fieldKindFrame -expand yes -fill both -padx 3 -pady 10
	pack $w.f.quantityFrame -expand yes -fill both
	pack $w.f.kernelsFrame -expand yes -fill both -pady 10
	
	makeSciButtonPanel $w $w $this
	moveToCursor $w
	
	pack $w.f -expand 1 -fill x
    }
}


