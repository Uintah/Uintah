ADD_LIBRARY(Packages_Uintah_Dataflow_Ports
	ArchivePort.cc 
	ScalarParticlesPort.cc 
	VectorParticlesPort.cc 
	TensorParticlesPort.cc 
)

TARGET_LINK_LIBRARIES(Packages_Uintah_Dataflow_Ports
	Dataflow_Network 
	Dataflow_Comm 
	Core_Containers 
        Core_Thread 
	Core_Geom 
	Core_Util 
	Core_Geometry 
	Core_Exceptions 
        Core_Persistent 
	Core_Datatypes 
	Core_GeomInterface 
	Packages_Uintah_Core_Grid 
	Packages_Uintah_Core_Datatypes 
        ${XML2_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_Dataflow_Ports)
ENDIF(BUILD_SHARED_LIBS)
