##
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




itcl_class CardioWave_Tools_CreateParametersBundle {
    inherit Module

    constructor {config} {
        set name CreateParametersBundle
        set_defaults
    }

    method set_defaults {} {


    global $this-par-names
    global $this-par-values
    global $this-par-types
    global $this-par-desceriptions
    global $this-types
    global $this-synchronise

    set $this-par-names  {{name}}
    set $this-par-values {{test}} 
    set $this-par-types {{string}}
    set $this-par-descriptions {{description}}
    set $this-synchronise "$this SynchroniseAll"

    set $this-types {{string} {double} {integer} {none}}
	}


    method updategui {} {

    global $this-par-names
    global $this-par-values
    global $this-par-types
    global $this-par-descriptions
    global $this-types
        
   	set w .ui[modname]
		if {[winfo exists $w]} {

            set parnames [set $this-par-names]
            set parvalues [set $this-par-values]
            set partypes [set $this-par-types]
            set pardescriptions [set $this-par-descriptions]            
 
            set numparams [llength $parnames]
            set childframe [$w.parametersframe childsite]
            pack forget $childframe.parframe
            destroy $childframe.parframe
            
            iwidgets::scrolledframe $childframe.parframe
            pack $childframe.parframe -fill both -expand yes
            set parframe [$childframe.parframe childsite]

            label $parframe.name -text "NAME"
            label $parframe.value -text "VALUE"
            label $parframe.type -text "TYPE"
	    label $parframe.description -text "DESCRIPTION"
            grid $parframe.name -column 0 -row 0
            grid $parframe.value -column 1 -row 0
            grid $parframe.type -column 3 -row 0
            grid $parframe.description -column 2 -row 0

            for {set x 0} {$x < $numparams} {incr x} {            
                set tname [lindex $parnames $x]
                set value [lindex $parvalues $x]
                set type [lindex $partypes $x]
		set description [lindex $pardescriptions $x]
                
                entry $parframe.name-$x -width 14
                bind  $parframe.name-$x <Leave> "$this SynchroniseName $x"
                bind  $parframe.name-$x <Key> "$this SynchroniseName $x"
                $parframe.name-$x insert end $tname
                grid $parframe.name-$x -column 0 -row [expr $x + 1] 
                entry $parframe.value-$x -width 12
                bind  $parframe.value-$x <Leave> "$this SynchroniseValue $x"
                bind  $parframe.value-$x <Key> "$this SynchroniseValue $x"
                $parframe.value-$x insert end $value
                grid $parframe.value-$x -column 1 -row [expr $x + 1] 

                iwidgets::optionmenu $parframe.type-$x -command "$this SynchroniseType $x"
                foreach d [set $this-types] {
                  $parframe.type-$x insert end $d
                }
                set index [lsearch [set $this-types] $type ]
                if [expr $index > 0] { $parframe.type-$x select $index }
                grid $parframe.type-$x -column 3 -row [expr $x + 1]

                entry $parframe.description-$x -width 35
                bind  $parframe.description-$x <Leave> "$this SynchroniseDescription $x"
                bind  $parframe.description-$x <Key> "$this SynchroniseDescription $x"
                $parframe.description-$x insert end $description
                grid $parframe.description-$x -column 2 -row [expr $x + 1] 

            }
          
            
           button $parframe.addnew -text "add parameter" -command "$this AddParameter"
           grid $parframe.addnew -column 0 -row [expr $x + 2] 
                
        return;
      }
    }

    method AddParameter {} {

    $this SynchroniseAll

    global $this-par-names
    global $this-par-values
    global $this-par-types
    global $this-par-descriptions
 
    lappend $this-par-names "name"
    lappend $this-par-values "value"
    lappend $this-par-types "string"
    lappend $this-par-descriptions "description"

    $this updategui
    
    }


    method ui {} {
       
  set w .ui[modname]
  if {[winfo exists $w]} {
        raise $w
        return;
  }

  global $this-par-names
  global $this-par-values
  global $this-par-types
  global $this-par-descriptions
 
  # create a new gui window

  toplevel $w 

  wm minsize $w 600 200
        
  iwidgets::labeledframe $w.parametersframe -labeltext "PARAMETERS"
  set childframe [$w.parametersframe childsite]
  pack $w.parametersframe -fill both -expand yes

  frame $childframe.parframe
  pack $childframe.parframe -fill both -expand yes
         
  $this updategui
    
  makeSciButtonPanel $w $w $this
  }

  
  method SynchroniseAll {} {
  
    global $this-par-names
    global $this-par-types
    global $this-par-values
    global $this-par-descriptions

    set w .ui[modname]

    if {[winfo exists $w]} {

            set numparams [llength [set $this-par-names]]

            set childframe [$w.parametersframe childsite]
            set parframe [$childframe.parframe childsite]
            
            for {set x 0} {$x < $numparams} {incr x} {  
              set value [$parframe.name-$x get]
              set $this-par-names [lreplace [set $this-par-names] $x $x $value]
              set value [$parframe.value-$x get]
              set $this-par-values [lreplace [set $this-par-values] $x $x $value]
              set value [$parframe.type-$x get]
              set $this-par-types [lreplace [set $this-par-types] $x $x $value]            
              set value [$parframe.description-$x get]
              set $this-par-descriptions [lreplace [set $this-par-descriptions] $x $x $value]            
  
            }
            
            
        }
  
  }



  method SynchroniseName {x} {

    global $this-par-names
    set w .ui[modname]

    if {[winfo exists $w]} {

            set childframe [$w.parametersframe childsite]
            set parframe [$childframe.parframe childsite]
            set value [$parframe.name-$x get]
            set $this-par-names [lreplace [set $this-par-names] $x $x $value]
        }
    }


  method SynchroniseValue {x} {

    global $this-par-values
    set w .ui[modname]

    if {[winfo exists $w]} {

            set childframe [$w.parametersframe childsite]
            set parframe [$childframe.parframe childsite]
            set value [$parframe.value-$x get]
            set $this-par-values [lreplace [set $this-par-values] $x $x $value]
        }
    }

  method SynchroniseType {x} {

    global $this-par-types
    set w .ui[modname]

    if {[winfo exists $w]} {

            set childframe [$w.parametersframe childsite]
            set parframe [$childframe.parframe childsite]
            set value [$parframe.type-$x get]
            set $this-par-types [lreplace [set $this-par-types] $x $x $value]
        }
  }

  method SynchroniseDescription {x} {

    global $this-par-descriptions
    set w .ui[modname]

    if {[winfo exists $w]} {

            set childframe [$w.parametersframe childsite]
            set parframe [$childframe.parframe childsite]
            set value [$parframe.description-$x get]
            set $this-par-descriptions [lreplace [set $this-par-descriptions] $x $x $value]
        }
  }


}
