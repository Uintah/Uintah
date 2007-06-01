#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# CMakeLists.txt for Packages/Uintah/StandAlone

ADD_EXECUTABLE(sus sus.cc)
ADD_EXECUTABLE(partvarRange partvarRange.cc)
ADD_EXECUTABLE(selectpart selectpart.cc)
ADD_EXECUTABLE(compare_uda compare_uda.cc)
ADD_EXECUTABLE(slb slb.cc)
ADD_EXECUTABLE(async_mpi_test async_mpi_test.cc)
ADD_EXECUTABLE(restart_merger restart_merger.cc)
ADD_EXECUTABLE(gambitFileReader gambitFileReader.cc)

SET(SUS_LIBS 
        Core_Containers   
        Core_Exceptions   
        Core_Geometry     
        Core_Math         
        Core_Persistent   
        Core_Thread       
        Core_Util         
        Packages_Uintah_Core_DataArchive 
        Packages_Uintah_Core_Grid        
        Packages_Uintah_Core_Parallel    
        Packages_Uintah_Core_Labels      
        Packages_Uintah_Core_Util        
        Packages_Uintah_Core_Math        
        Packages_Uintah_Core_Disclosure  
        Packages_Uintah_Core_Exceptions  
        Packages_Uintah_CCA_Ports        
        Packages_Uintah_CCA_Components_Parent 
        Packages_Uintah_CCA_Components_Models 
        Packages_Uintah_CCA_Components_DataArchiver  
        Packages_Uintah_CCA_Components_LoadBalancers 
        Packages_Uintah_CCA_Components_Regridder     
        Packages_Uintah_Core_ProblemSpec             
        Packages_Uintah_CCA_Components_SimulationController 
        Packages_Uintah_CCA_Components_Schedulers           
        Packages_Uintah_CCA_Components_ProblemSpecification 
        Packages_Uintah_CCA_Components_Solvers
)

IF(HYPRE_LIBRARY)
  SET(SUS_LIBS ${SUS_LIBS} ${HYPRE_LIBRARY})
ENDIF(HYPRE_LIBRARY)
IF(PETSC_LIBRARY)
  SET(SUS_LIBS ${SUS_LIBS} ${PETSC_LIBRARY})
ENDIF(PETSC_LIBRARY)

TARGET_LINK_LIBRARIES(sus ${SUS_LIBS})

TARGET_LINK_LIBRARIES(compare_uda
  Packages_Uintah_Core_DataArchive
  Packages_Uintah_Core_Grid
  Core_Util
  Core_Thread
  Core_Math
)

TARGET_LINK_LIBRARIES(partvarRange
  Core_Containers
  Core_Exceptions
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Disclosure
  Packages_Uintah_Core_DataArchive
)

TARGET_LINK_LIBRARIES(async_mpi_test
  ${MPI_LIBRARY}
)

TARGET_LINK_LIBRARIES(restart_merger
  Core_Exceptions
  Core_OS
  Packages_Uintah_Core_DataArchive
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_CCA_Components_DataArchiver
)
  
TARGET_LINK_LIBRARIES(selectpart
  Core_Containers
  Core_Exceptions
  Core_Geometry
  Core_Util
  Packages_Uintah_Core_DataArchive
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_Util
)
  
TARGET_LINK_LIBRARIES(slb
  Core_Exceptions
  Packages_Uintah_Core_GeometryPiece
  Packages_Uintah_Core_Grid
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_CCA_Components_ProblemSpecification
)

IF(TEEM_LIBRARY AND !MINIMAL_CORE)  
  ADD_EXECUTABLE(uda2nrrd uda2nrrd.cc)
  TARGET_LINK_LIBRARIES(uda2nrrd
    Core_Exceptions
    Core_Datatypes
    Core_Geometry
    Packages_Uintah_Core_DataArchive
    Packages_Uintah_Core_Grid
  )
ENDIF(TEEM_LIBRARY AND !MINIMAL_CORE)  
  
  

ADD_SUBDIRECTORY(tools)
ADD_SUBDIRECTORY(Benchmarks)
