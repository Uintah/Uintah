#ifndef UDA2NRRD_UPDATE_MESH_HANDLE_H
#define UDA2NRRD_UPDATE_MESH_HANDLE_H

#include <SCIRun/Core/Basis/HexTrilinearLgn.h>
#include <SCIRun/Core/Datatypes/LatVolMesh.h>

#include <Core/Grid/Level.h>
#include <Core/Grid/LevelP.h>
#include <Core/Disclosure/TypeDescription.h>

#include <StandAlone/tools/uda2nrrd/Args.h>

#include <SCIRun/Core/Geometry/IntVector.h>
#include <SCIRun/Core/Geometry/Point.h>
#include <SCIRun/Core/Geometry/BBox.h>

typedef SCIRun::LatVolMesh<SCIRun::HexTrilinearLgn<SCIRun::Point> > LVMesh;
typedef LVMesh::handle_type LVMeshHandle;

bool update_mesh_handle( Uintah::LevelP& level,
                         SCIRun::IntVector& hi,
                         SCIRun::IntVector& range,
                         SCIRun::BBox& box,
                         Uintah::TypeDescription::Type type,
                         LVMeshHandle& mesh_handle,
                         const Args & args );

#endif // UDA2NRRD_UPDATE_MESH_HANDLE_H


