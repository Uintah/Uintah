########################################
#CLASS
#    VizControl
#    Visualization control for simulation data that contains
#    information on both a regular grid in particle sets.
#OVERVIEW TEXT
#    This module receives a ParticleGridReader object.  The user
#    interface is dynamically created based information provided by the
#    ParticleGridReader.  The user can then select which variables he/she
#    wishes to view in a visualization.
#KEYWORDS
#    ParticleGridReader, Material/Particle Method
#AUTHOR
#    Kurt Zimmerman
#    Department of Computer Science
#    University of Utah
#    January 1999
#    Copyright (C) 1999 SCI Group
#LOG
#    Created January 5, 1999
########################################

catch {rename ScalarFieldExtractor ""}

itcl_class Uintah_Selectors_ScalarFieldExtractor { 
    inherit Uintah_Selectors_FieldExtractor 

    constructor {config} { 
        set name ScalarFieldExtractor
	set label_text "Scalar Fields"
        set_defaults
    } 
}    
	    	    
