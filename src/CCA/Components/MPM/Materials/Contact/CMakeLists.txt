set( MPM_SRC ${MPM_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ApproachContact.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/CompositeContact.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Contact.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ContactFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ContactMaterialSpec.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/FrictionContactBard.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/FrictionContactLR.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/NodalSVFContact.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/NullContact.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SingleVelContact.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SpecifiedBodyContact.cc
        PARENT_SCOPE
    )
