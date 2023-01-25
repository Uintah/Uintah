set( ICE_SRC ${ICE_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/AdvectionFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Advector.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/FirstOrderAdvector.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SecondOrderAdvector.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/SecondOrderBase.cc
        PARENT_SCOPE
    )
set( ICE_INC ${ICE_INC}
        ${CMAKE_CURRENT_SOURCE_DIR}
        PARENT_SCOPE
    )