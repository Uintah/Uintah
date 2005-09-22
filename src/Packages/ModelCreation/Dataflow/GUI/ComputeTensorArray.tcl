itcl_class ModelCreation_TensorVectorMath_ComputeTensorArray {
    inherit Module
    constructor {config} {
        set name ComputeTensorArray
        set_defaults
    }

    method set_defaults {} {
      global $this-function
      global $this-help

      set $this-function "RESULT = eig1(A);"
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

        iwidgets::labeledframe $w.inf -labeltext "Create Tensor Array"
        set infoframe [$w.inf childsite]
        frame $infoframe.info
        pack $infoframe.info -side left
        set info $infoframe.info
        label $info.info1 -text "Function: RESULT = function(A,B,C,...)"
        label $info.info2 -text "Input array: A, B, C, ... (scalar,vector, or tensor)"
        label $info.info3 -text "Output array: RESULT (tensor)"
        label $info.info4 -text "Element index: INDEX (scalar)"
        label $info.info5 -text "Number of elements: SIZE (scalar)"
        grid $info.info1 -row 0 -column 0 -sticky w
        grid $info.info2 -row 1 -column 0 -sticky w
        grid $info.info3 -row 2 -column 0 -sticky w
        grid $info.info4 -row 3 -column 0 -sticky w
        grid $info.info5 -row 4 -column 0 -sticky w
        pack $w.inf -side top -anchor w -fill x
        
        iwidgets::labeledframe $w.ff -labeltext "function"
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
              $this-c gethelp
        $help.help render [set $this-help]
        pack $w.hf -side top -anchor w -fill both -expand yes
        pack $help.help -side top -fill both -expand yes

        makeSciButtonPanel $w $w $this
        moveToCursor $w
    }


}


