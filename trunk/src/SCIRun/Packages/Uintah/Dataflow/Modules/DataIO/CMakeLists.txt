ADD_LIBRARY(Packages_Uintah_Dataflow_Modules_DataIO ArchiveReader.cc)

TARGET_LINK_LIBRARIES(Packages_Uintah_Dataflow_Modules_DataIO
	Packages_Uintah_Core_Datatypes   
	Packages_Uintah_Core_DataArchive 
	Packages_Uintah_Core_ProblemSpec 
	Packages_Uintah_Core_Math 
	Packages_Uintah_Core_Grid 
	Packages_Uintah_CCA_Ports 
	Dataflow_GuiInterface  
	Dataflow_Network   
	Core_Containers    
	Core_Datatypes     
	Core_Exceptions    
	Core_Geom          
	Core_GeomInterface 
	Core_Persistent    
	Core_Thread       
	${XML2_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_Dataflow_Modules_DataIO)
ENDIF(BUILD_SHARED_LIBS)
