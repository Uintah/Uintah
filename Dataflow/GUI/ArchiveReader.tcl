itcl_class Uintah_DataIO_ArchiveReader { 

    inherit Module 
    
    constructor {config} { 
        set name ArchiveReader 
        set_defaults
    } 
  	
    method set_defaults {} { 
	global $this-filebase 
	set $this-filebase ""
    } 
  

    method ui {} { 

	global $this-filebase
	set $this-filebase [tk_chooseDirectory -parent . -mustexist 1 \
				-initialdir [set $this-filebase] ]
	$this-c needexecute
    }
}
