
#include <Classlib/Queue.h>

typedef Queue<int> hack1;

#include <Classlib/Array3.h>
#include <Geometry/Vector.h>
typedef Array3<Vector> hack2;

#include <Datatypes/Mesh.h>
typedef MeshHandle hack3;

#include <Datatypes/TriSurface.h>
#include <Classlib/Array1.h>
typedef Array1<TSElement*> hack4;

#include <Datatypes/ContourSet.h>
typedef ContourSetHandle hack5;

#include <Datatypes/ScalarField.h>
typedef ScalarFieldHandle hack6;

#include <Datatypes/VectorField.h>
typedef VectorFieldHandle hack7;

#include <Datatypes/GeometryPort.h>
typedef Mailbox<GeomReply> hack8;

#include <Datatypes/MeshPort.h>
typedef MeshIPort hack9;

#include <Datatypes/Matrix.h>
typedef MatrixHandle hack10;
