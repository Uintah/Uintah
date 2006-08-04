
catch {rename Span ""}

itcl_class Yarden_Visualization_Span {
    inherit Module
    constructor {config} {
	set name Span
	set_defaults
    }
    method set_defaults {} {
	global $this-split
	global $this-max_split
	global $this-z_max
	global $this-z_from 
	global $this-z_size

	set $this-split 1
	set $this-max_split 32
	set $this-z_max 64
	set $this-z_from 0
	set $this-z_size 64
    }
    method ui {} {
	global $this-split
	global $this-max_split
	global $this-z_max
	global $this-z_from
	global $this-z_size

	set w .ui[modname]
	puts "scan ui : $w"
	if {[winfo exists $w]} {
	    raise $w
	    return;
	}

	toplevel $w
	frame $w.f 
	pack $w.f -padx 2 -pady 2 -fill x
	
	button $w.f.split -text "Split" -command "$this-c needexecute"

 	scale $w.f.np  \
	    -label "proc" \
 	    -variable $this-split \
 	    -from 0 -to [set $this-max_split] \
 	    -length 5c \
 	    -showvalue true \
 	    -orient horizontal  
	
	pack $w.f.split -side top
	pack $w.f.np -side top  -fill x

 	frame $w.f.z
 	pack $w.f.z -side top

 	scale $w.f.z.from  \
	    -label "From:" \
 	    -length 15c \
 	    -variable $this-z_from \
 	    -from 0 -to [set $this-z_max] \
 	    -showvalue true \
 	    -orient horizontal

 	scale $w.f.z.size  \
	    -label "Size" \
 	    -variable $this-z_size \
 	    -length 15c \
 	    -from 0 -to [set $this-z_max] \
 	    -orient horizontal \
 	    -showvalue true 

 	pack $w.f.z.from $w.f.z.size -side top  -fill x

  	trace variable $this-max_split w "$this change_max_split"
 	trace variable $this-z_max w "$this change_z_max"
   }

   method change_max_split {n1 n2 op} {
       global $this-max_split
       .ui$this.f.split  configure -to [set $this-max_split]
       puts "change_split [set $this-max_split]"
   }

     method change_z_max {n1 n2 op} {
 	global $this-z_max
 	.ui$this.f.z.from  configure -to [set $this-z_max]
 	.ui$this.f.z.size  configure -to [set $this-z_max]
     }
}
