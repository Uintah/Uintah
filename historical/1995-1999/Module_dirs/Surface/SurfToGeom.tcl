
catch {rename SurfToGeom ""}

itcl_class SurfToGeom {
    inherit Module
    constructor {config} {
	set name SurfToGeom
	set_defaults
    }
    method set_defaults {} {
    }
}
