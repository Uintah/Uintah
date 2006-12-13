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
PROJECT(SCIRUN_STANDALONE_CONVERT)

SET (EXECUTABLE_OUTPUT_PATH ${SCIRUN_BINARY_DIR}/bin/convert CACHE INTERNAL "Single output directory for building all executables.")
#output directory for installing all include files.")
MARK_AS_ADVANCED(EXECUTABLE_OUTPUT_PATH)

SET(StandAlone_Convert_Libs
  Core_Datatypes
  Core_Util
  Core_Containers
  Core_Persistent
  Core_Util
  Core_Exceptions
  Core_Init
  Core_Thread
  Core_Geometry
  Core_Math
  Core_Geom
  Core_ImportExport
  Core_Basis
  ${XML2_LIBRARY}
  ${M_LIBRARY}
  ${F_LIBRARY}
)

ADD_EXECUTABLE(CurveFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/CurveFieldToText.cc)
TARGET_LINK_LIBRARIES(CurveFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(HexVolFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/HexVolFieldToText.cc)
TARGET_LINK_LIBRARIES(HexVolFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(HexVolFieldToExotxt ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/HexVolFieldToExotxt.cc)
TARGET_LINK_LIBRARIES(HexVolFieldToExotxt ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(HexVolToVtk ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/HexVolToVtk.cc)
TARGET_LINK_LIBRARIES(HexVolToVtk ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(PointCloudFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/PointCloudFieldToText.cc)
TARGET_LINK_LIBRARIES(PointCloudFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(PointCloudFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/PointCloudFieldToText.cc)
TARGET_LINK_LIBRARIES(PointCloudFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(StructCurveFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/StructCurveFieldToText.cc)
TARGET_LINK_LIBRARIES(StructCurveFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(StructHexVolFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/StructHexVolFieldToText.cc)
TARGET_LINK_LIBRARIES(StructHexVolFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(StructQuadSurfFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/StructQuadSurfFieldToText.cc)
TARGET_LINK_LIBRARIES(StructQuadSurfFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TetVolFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TetVolFieldToText.cc)
TARGET_LINK_LIBRARIES(TetVolFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TriSurfFieldToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TriSurfFieldToText.cc)
TARGET_LINK_LIBRARIES(TriSurfFieldToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TriSurfFieldToExotxt ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TriSurfFieldToExotxt.cc)
TARGET_LINK_LIBRARIES(TriSurfFieldToExotxt ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(ColumnMatrixToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/ColumnMatrixToText.cc)
TARGET_LINK_LIBRARIES(ColumnMatrixToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(DenseMatrixToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/DenseMatrixToText.cc)
TARGET_LINK_LIBRARIES(DenseMatrixToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(SparseRowMatrixToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/SparseRowMatrixToText.cc)
TARGET_LINK_LIBRARIES(SparseRowMatrixToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(ColorMapToText ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/ColorMapToText.cc)
TARGET_LINK_LIBRARIES(ColorMapToText ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToCurveField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToCurveField.cc)
TARGET_LINK_LIBRARIES(TextToCurveField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToHexVolField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToHexVolField.cc)
TARGET_LINK_LIBRARIES(TextToHexVolField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToHexTricubicHmt ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToHexTricubicHmt.cc)
TARGET_LINK_LIBRARIES(TextToHexTricubicHmt ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToPointCloudField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToPointCloudField.cc)
TARGET_LINK_LIBRARIES(TextToPointCloudField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToPointCloudString ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToPointCloudString.cc)
TARGET_LINK_LIBRARIES(TextToPointCloudString ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToQuadSurfField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToQuadSurfField.cc)
TARGET_LINK_LIBRARIES(TextToQuadSurfField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToStructCurveField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToStructCurveField.cc)
TARGET_LINK_LIBRARIES(TextToStructCurveField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToStructHexVolField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToStructHexVolField.cc)
TARGET_LINK_LIBRARIES(TextToStructHexVolField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToStructQuadSurfField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToStructQuadSurfField.cc)
TARGET_LINK_LIBRARIES(TextToStructQuadSurfField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToTetVolField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToTetVolField.cc)
TARGET_LINK_LIBRARIES(TextToTetVolField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToTriSurfField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToTriSurfField.cc)
TARGET_LINK_LIBRARIES(TextToTriSurfField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(MtoTriSurfField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/MtoTriSurfField.cc)
TARGET_LINK_LIBRARIES(MtoTriSurfField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(VTKtoTriSurfField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/VTKtoTriSurfField.cc)
TARGET_LINK_LIBRARIES(VTKtoTriSurfField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(VTKtoHexVolField ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/VTKtoHexVolField.cc)
TARGET_LINK_LIBRARIES(VTKtoHexVolField ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToColumnMatrix ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToColumnMatrix.cc)
TARGET_LINK_LIBRARIES(TextToColumnMatrix ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToDenseMatrix ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToDenseMatrix.cc)
TARGET_LINK_LIBRARIES(TextToDenseMatrix ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TextToColorMap ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TextToColorMap.cc)
TARGET_LINK_LIBRARIES(TextToColorMap ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TriSurfToVtk ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TriSurfToVtk.cc)
TARGET_LINK_LIBRARIES(TriSurfToVtk ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TriSurfToOBJ ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TriSurfToOBJ.cc)
TARGET_LINK_LIBRARIES(TriSurfToOBJ ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TriSurfToTetgen ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TriSurfToTetgen.cc)
TARGET_LINK_LIBRARIES(TriSurfToTetgen ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TriSurfToM ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TriSurfToM.cc)
TARGET_LINK_LIBRARIES(TriSurfToM ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(TetgenToTetVol ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/TetgenToTetVol.cc)
TARGET_LINK_LIBRARIES(TetgenToTetVol ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(vff2nrrd ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/vff2nrrd.cc)
TARGET_LINK_LIBRARIES(vff2nrrd ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(PictToNrrd ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/PictToNrrd.cc)
TARGET_LINK_LIBRARIES(PictToNrrd ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(UCSDToHexVol ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/UCSDToHexVol.cc)
TARGET_LINK_LIBRARIES(UCSDToHexVol ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(UCSD_HO_edge_sfToHexVol ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/UCSD_HO_edge_sfToHexVol.cc)
TARGET_LINK_LIBRARIES(UCSD_HO_edge_sfToHexVol ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(VistaToNrrd ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/VistaToNrrd.cc)
TARGET_LINK_LIBRARIES(VistaToNrrd ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(gmvToPts ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/gmvToPts.cc)
TARGET_LINK_LIBRARIES(gmvToPts ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(geoToPts ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/geoToPts.cc)
TARGET_LINK_LIBRARIES(geoToPts ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(SampleHexTricubicHmt ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/SampleHexTricubicHmt.cc)
TARGET_LINK_LIBRARIES(SampleHexTricubicHmt ${StandAlone_Convert_Libs})

ADD_EXECUTABLE(Lin2QuadMesh ${SCIRUN_CORE_SOURCE_DIR}/StandAlone/convert/Lin2QuadMesh.cc)
TARGET_LINK_LIBRARIES(Lin2QuadMesh ${StandAlone_Convert_Libs})
