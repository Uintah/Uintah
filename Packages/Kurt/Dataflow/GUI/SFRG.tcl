
catch {rename SFRG ""}

itcl_class Kurt_Vis_SFRG {
    inherit Dataflow_Fields_SFRGfile
    constructor {config} {
	set name SFRG
	set_defaults
    }
}
