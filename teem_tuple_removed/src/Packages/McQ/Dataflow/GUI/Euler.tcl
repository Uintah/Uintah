itcl_class Packages/McQ_SAMRAI_Euler {
  inherit Module

  constructor {config} {
    set name "SAMRAI Euler"
  }

  method ui {} {
    set w .ui$this
    if {[winfo exists $w]} {
      raise $w
      return;
    }
    toplevel $w
    wm minsize $w 300 20
  }
}
