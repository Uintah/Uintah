add_library( CCA_Components_Models ModelFactory.cc )

# jcs need to plug in all of the ICE/MPM specific models

target_link_libraries( CCA_Components_Models
        CCA_Ports
        Core_Exceptions
        Core_ProblemSpec
    )