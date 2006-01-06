itcl_class ModelCreation_FieldsCreate_ClipFieldByFunction {
    inherit Module
    constructor {config} {
        set name ClipFieldByFunction
        set_defaults
    }

   method set_defaults {} {
      global $this-function
      global $this-format
      global $this-help

      set $this-function "DATA < A;"
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
        $this-c gethelp

        toplevel $w

        iwidgets::labeledframe $w.inf -labeltext "Create Clipping Expression"
        set infoframe [$w.inf childsite]
        frame $infoframe.info
        pack $infoframe.info -side left
        set info $infoframe.info
        label $info.info1 -text "Function: expression(DATA,A,B,C,...)"
        label $info.info2 -text "Input array: DATA (scalar/vector/tensor: data from field port) "
        label $info.info3 -text "Input array: X, Y, Z (scalar: Cartensian coordinates of node/element)"
        label $info.info4 -text "Input array: POS (vector: vector with node/element position)"
        label $info.info5 -text "Input array: A, B, C, ... (scalar/vector/tensor: data from matrix ports)"
        label $info.info6 -text "Input array: INDEX (scalar: number of the element)"
        label $info.info7 -text "Input array: SIZE (scalar: number of elements)"
        label $info.info8 -text "Input array: ELEMENT (element: object containing element properties)"

        grid $info.info1 -row 0 -column 0 -columnspan 2 -sticky w
        grid $info.info2 -row 1 -column 0 -sticky w
        grid $info.info3 -row 2 -column 0 -sticky w
        grid $info.info4 -row 3 -column 0 -sticky w
        grid $info.info5 -row 4 -column 0 -sticky w
        grid $info.info6 -row 1 -column 1 -sticky w
        grid $info.info7 -row 2 -column 1 -sticky w
        grid $info.info8 -row 3 -column 1 -sticky w

        pack $w.inf -side top -anchor w -fill x

        iwidgets::labeledframe $w.ff -labeltext "expression"
        set function [$w.ff childsite]
        option add *textBackground white	
        iwidgets::scrolledtext $function.function -height 60 -hscrollmode dynamic
        $function.function insert end [set $this-function]
        pack $w.ff -side top -anchor w -fill both 
        pack $function.function -side top -fill both 

        iwidgets::labeledframe $w.hf -labeltext "available functions"
        set help [$w.hf childsite]
        option add *textBackground white	
        iwidgets::scrolledhtml $help.help -height 60 -hscrollmode dynamic
        pack $w.hf -side top -anchor w -fill both -expand yes
        pack $help.help -side top -fill both -expand yes

        $help.help render [set $this-help]

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }

}


