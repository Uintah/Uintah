# Sources of Core/Datatypes classes
SET(Core_Datatypes_SRCS
        Clipper.cc
	Color.cc		    	
	ColumnMatrix.cc	    	
	Datatype.cc		    	
	DenseColMajMatrix.cc	    	
	DenseMatrix.cc	    	
	Field.cc		    	
	FieldInterfaceAux.cc	    	
	Image.cc		    	
	Matrix.cc		    	
	MatrixOperations.cc	    	
	Mesh.cc		    	
	NrrdData.cc		    	
	PropertyManager.cc	    	
	SearchGrid.cc           	
	SparseRowMatrix.cc	    	
        String.cc                     
	TypeName.cc		    	
	cd_templates.cc	    	
	cd_templates_fields_0.cc    	
	cd_templates_fields_1.cc    	
	cd_templates_fields_2.cc    	
	cd_templates_fields_3.cc    	
	cd_templates_fields_4.cc    	
	cd_templates_fields_5.cc	
	cd_templates_fields_6.cc	
)

IF(HAVE_INSIGHT)
  SET(Core_Datatypes_SRCS ${Core_Datatypes_SRCS} 
    ITKDatatype.cc
    cd_templates_itk.cc
    cd_templates_fields_0_itk.cc
    cd_templates_fields_1_itk.cc)
ENDIF(HAVE_INSIGHT)

ADD_LIBRARY(Core_Datatypes ${Core_Datatypes_SRCS})

TARGET_LINK_LIBRARIES(Core_Datatypes 
  Core_Basis        
  Core_Containers   
  Core_Exceptions   
  Core_Geometry     
  Core_Math         
  Core_Persistent   
  Core_Thread       
  Core_Util
  ${GL_LIBRARY}
  ${M_LIBRARY}
  ${BLAS_LIBRARY}
  ${F_LIBRARY} 
  ${TEEM_LIBRARY}
  ${PNG_LIBRARY}
  ${Z_LIBRARY}
)

IF(HAVE_INSIGHT)
  TARGET_LINK_LIBRARIES(Core_Datatypes
    ITKCommon)
ENDIF(HAVE_INSIGHT)

IF(NOT BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Core_Datatypes)
ENDIF(NOT BUILD_SHARED_LIBS)
