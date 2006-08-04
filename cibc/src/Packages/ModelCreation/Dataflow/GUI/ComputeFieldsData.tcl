itcl_class ModelCreation_FieldsData_ComputeFieldsData {
   inherit Module

    constructor {config} {
        set name ComputeFieldsData
        set_defaults
    }

    method set_defaults {} {
      global $this-function
      global $this-format
      global $this-help

      set $this-function "RESULT = abs(DATA);"
      set $this-format "Scalar"
      set $this-help ""
      
    }

    method update_text {} {
      set w .ui[modname]
      if {[winfo exists $w]} {
        set function [$w.ff childsite]
        set $this-function [$function.function get 1.0 end]
        }
    }

    method ui {} {
        set w .ui[modname]
        if {[winfo exists $w]} {
            return
        }

        toplevel $w

        iwidgets::labeledframe $w.inf -labeltext "Create New Field Data"
        set infoframe [$w.inf childsite]
        frame $infoframe.info
        pack $infoframe.info -side left
        set info $infoframe.info
        label $info.info1 -text "Function: RESULT = function(DATA,A,B,C,...)"
        label $info.info2  -text "Input array: DATA1 (scalar/vector/tensor: data from field port) "
        label $info.info2a -text "Input array: DATA2 (scalar/vector/tensor: data from field port) "
        label $info.info2b -text "Input array: DATA3 (scalar/vector/tensor: data from field port) "
        label $info.info3 -text "Input array: X, Y, Z (scalar: Cartensian coordinates of node/element)"
        label $info.info4 -text "Input array: POS (vector: vector with node/element position)"
        label $info.info5 -text "Input array: A, B, C, ... (scalar/vector/tensor: data from field data ports)"
        label $info.info6 -text "Input array: INDEX (scalar: number of the element)"
        label $info.info7 -text "Input array: SIZE (scalar: number of elements)"
        label $info.info8 -text "Input array: ELEMENT (element: object containing element properties)"
        label $info.info9 -text "Output array: RESULT (scalar)"

        grid $info.info1 -row 0 -column 0 -columnspan 2 -sticky w
        grid $info.info2 -row 1 -column 0 -sticky w
        grid $info.info2a -row 2 -column 0 -sticky w        
        grid $info.info2b -row 3 -column 0 -sticky w
        grid $info.info3 -row 4 -column 0 -sticky w
        grid $info.info4 -row 5 -column 0 -sticky w
        grid $info.info5 -row 1 -column 1 -sticky w
        grid $info.info6 -row 2 -column 1 -sticky w
        grid $info.info7 -row 3 -column 1 -sticky w
        grid $info.info8 -row 4 -column 1 -sticky w
        grid $info.info9 -row 5 -column 1 -sticky w

        pack $w.inf -side top -anchor w -fill x

        iwidgets::labeledframe $w.of -labeltext "output type"
        set otype [$w.of childsite]
        pack $w.of -side top -anchor w -fill x
        
        labelcombo $otype.otype "Field Output Data Type" \
          {Scalar Vector Tensor "Same as Input"} \
          $this-format

        iwidgets::labeledframe $w.ff -labeltext "function"
        set function [$w.ff childsite]
        option add *textBackground white	
        iwidgets::scrolledtext $function.function -height 60 -hscrollmode dynamic
        $function.function insert end [set $this-function]
        bind $function.function <Leave> "$this update_text"
                
        pack $w.ff -side top -anchor w -fill both 
        pack $function.function -side top -fill both 

        button $w.help -text "Available Functions" -command "$this showhelp"
        pack $w.help -side top -anchor e 

        makeSciButtonPanel $w $w $this
    }


    method showhelp { } {

      # Create a unique name for the file selection window
      set w [format "%s-functionhelp" .ui[modname]]

      if { [winfo exists $w] } {
        if { [winfo ismapped $w] == 1} {
          raise $w
        } else {
          wm deiconify $w
        }
	    	return
      }
	
      toplevel $w -class TkFDialog

      global $this-help
      $this-c gethelp   
            
      iwidgets::labeledframe $w.hf -labeltext "available functions"
      set help [$w.hf childsite]
      option add *textBackground white	
      iwidgets::scrolledhtml $help.help -height 60 -hscrollmode dynamic -width 500p -height 300p        
      $help.help render [set $this-help]
      pack $help.help -side top -anchor w -fill both -expand yes
      pack $w.hf -side top -anchor w -fill both -expand yes
    }    
 
    
    method labelcombo { win text1 arglist var} {
	frame $win 
	pack $win -side top -padx 5
	label $win.l1 -text $text1 \
		      -anchor w -just left
	label $win.colon  -text ":" -width 2 -anchor w -just left
	iwidgets::optionmenu $win.c -foreground darkred \
		-command " $this comboget $win.c $var "

	set i 0
	set found 0
	set length [llength $arglist]
	for {set elem [lindex $arglist $i]} {$i<$length} \
	    {incr i 1; set elem [lindex $arglist $i]} {
	    if {"$elem"=="[set $var]"} {
		set found 1
	    }
	    $win.c insert end $elem
	}

	if {!$found} {
	    $win.c insert end [set $var]
	}

	$win.c select [set $var]

	label $win.l2 -text "" -width 20 -anchor w -just left

	# hack to associate optionmenus with a textvariable
	# bind $win.c <Map> "$win.c select {[set $var]}"

	pack $win.l1 $win.colon -side left
	pack $win.c $win.l2 -side left	
    }

    method comboget { win var } {
	if {![winfo exists $win]} {
	    return
	}
	if { "$var"!="[$win get]" } {
	    set $var [$win get]
	}
    }

    method set_combobox { win var name1 name2 op } {
	set w .ui[modname]
	set menu $w.$win
	if {[winfo exists $menu]} {
	    $menu select $var
	}
    }

}


