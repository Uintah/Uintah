
catch {rename GenColormap ""}

itcl_class GenColormap {
    inherit Module
    constructor {config} {
	set name GenColormap
	set_defaults
    }
    method set_defaults {} {
	global $this-nlevels
	set $this-nlevels 50
	
	global $this-ambient_intens
	set $this-ambient_intens 0
	
	global $this-diffuse_intens
	set $this-diffuse_intens 1

	global $this-specular_intens
	set $this-specular_intens 0

	global $this-spec_percent
	set $this-spec_percent 1

	global $this-spec_color-r $this-spec_color-g $this-spec_color-b
	set $this-spec_color-r .h
	set $this-spec_color-g .h
	set $this-spec_color-b .h
	
	global $this-shininess
	set $this-shininess 10

	global $this-map_type
	set $this-map_type rainbow

	global $this-rainbow_hue_min
	set $this-rainbow_hue_min 0
	global $this-rainbow_hue_max
	set $this-rainbow_hue_max 300
	global $this-rainbow_sat
	set $this-rainbow_sat 1
	global $this-rainbow_val
	set $this-rainbow_val 1
    }
}
