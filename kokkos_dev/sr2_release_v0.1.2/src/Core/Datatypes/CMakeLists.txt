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



ADD_LIBRARY(Core_Datatypes ${Core_Datatypes_SRCS})
TARGET_LINK_LIBRARIES(Core_Datatypes Core_Persistent)