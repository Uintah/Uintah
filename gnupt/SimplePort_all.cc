
#pragma implementation "SimplePort.h"

#include <Datatypes/SimplePort.cc>
#include <Datatypes/ContourSet.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/Surface.h>
#include <Datatypes/VectorField.h>

typedef SimpleIPort<ContourSetHandle> _dummy1_;
typedef SimpleOPort<ContourSetHandle> _dummy2_;
typedef SimpleIPort<ScalarFieldHandle> _dummy3_;
typedef SimpleOPort<ScalarFieldHandle> _dummy4_;
typedef SimpleIPort<SurfaceHandle> _dummy5_;
typedef SimpleOPort<SurfaceHandle> _dummy6_;
typedef SimpleIPort<VectorFieldHandle> _dummy7_;
typedef SimpleOPort<VectorFieldHandle> _dummy8_;
typedef SimpleIPort<MeshHandle> _dummy9_;
typedef SimpleOPort<MeshHandle> _dummy10_;
typedef SimpleIPort<MatrixHandle> _dummy11_;
typedef SimpleOPort<MatrixHandle> _dummy12_;
