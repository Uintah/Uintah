if( ENABLE_ICE AND ENABLE_MPM )
    set( MDL_SRC ${MDL_SRC}
        ${CMAKE_CURRENT_SOURCE_DIR}/ExchangeCoefficients.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ExchangeFactory.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/ExchangeModel.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Scalar.cc
        ${CMAKE_CURRENT_SOURCE_DIR}/Slip.cc
        PARENT_SCOPE
    )
endif()
