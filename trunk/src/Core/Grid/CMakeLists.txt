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

# CMakeLists.txt for Packages/Uintah/Core/Grid

SET(Packages_Uintah_Core_Grid_SRCS
  Box.h
  Box.cc
  Grid.cc
  Level.cc
  Material.cc
  PatchRangeTree.cc
  Region.cc
  Patch.cc
  Ghost.cc
  SimulationState.cc
  SimulationTime.cc
  Task.cc
  UnknownVariable.cc
  SimpleMaterial.cc
  LinearInterpolator.cc
  AMR.cc
  Node27Interpolator.cc
  BSplineInterpolator.cc
  TOBSplineInterpolator.cc

  Variables/CellIterator.cc               
  Variables/ComputeSet.cc                 
  Variables/ComputeSet_special.cc         
  Variables/GridIterator.cc               
  Variables/GridVariableBase.cc             
  Variables/LocallyComputedPatchVarMap.cc 
  Variables/ParticleSubset.cc             
  Variables/ParticleVariableBase.cc       
  Variables/ParticleVariable_special.cc   
  Variables/PerPatchBase.cc               
  Variables/PSPatchMatlGhost.cc           
  Variables/ReductionVariableBase.cc      
  Variables/ReductionVariable_special.cc  
  Variables/ScrubItem.cc                  
  Variables/SoleVariableBase.cc           
  Variables/SoleVariable_special.cc       
  Variables/Stencil7.cc                   
  Variables/ugc_templates.cc              
  Variables/VarLabel.cc                   
  Variables/Variable.cc

  BoundaryConditions/VelocityBoundCond.cc 
  BoundaryConditions/TemperatureBoundCond.cc
  BoundaryConditions/PressureBoundCond.cc
  BoundaryConditions/DensityBoundCond.cc
  BoundaryConditions/SpecificVolBoundCond.cc
  BoundaryConditions/MassFracBoundCond.cc
  BoundaryConditions/BoundCondFactory.cc
  BoundaryConditions/BoundCondReader.cc
  BoundaryConditions/BCData.cc
  BoundaryConditions/BCDataArray.cc
  BoundaryConditions/BCGeomBase.cc
  BoundaryConditions/UnionBCData.cc
  BoundaryConditions/DifferenceBCData.cc
  BoundaryConditions/SideBCData.cc
  BoundaryConditions/CircleBCData.cc
  BoundaryConditions/AnnulusBCData.cc
  BoundaryConditions/RectangleBCData.cc
                   
)

ADD_LIBRARY(Packages_Uintah_Core_Grid ${Packages_Uintah_Core_Grid_SRCS})

TARGET_LINK_LIBRARIES(Packages_Uintah_Core_Grid
  Core_Geometry
  Core_Exceptions
  Core_Thread
  Core_Util
  Core_Containers
  Packages_Uintah_Core_Malloc
  Packages_Uintah_Core_Parallel
  Packages_Uintah_Core_ProblemSpec
  Packages_Uintah_Core_Exceptions
  Packages_Uintah_Core_Util
  Packages_Uintah_Core_Math
  Packages_Uintah_Core_Disclosure
  ${MPI_LIBS}
  ${Z_LIBRARY}
)

IF(BUILD_SHARED_LIBS)
  ADD_DEFINITIONS(-DBUILD_Packages_Uintah_Core_Grid)
ENDIF(BUILD_SHARED_LIBS)

