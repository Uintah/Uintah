#ifndef UDA2NRRD_UPDATE_MESH_HANDLE_H
#define UDA2NRRD_UPDATE_MESH_HANDLE_H

#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>

#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>

#include <Packages/Uintah/StandAlone/tools/uda2vis/Args.h>

class SCIRun::BBox;
class SCIRun::IntVector;
class SCIRun::Point;

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


