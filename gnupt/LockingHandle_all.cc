
#pragma implementation "LockingHandle.h"

#include <Classlib/LockingHandle.cc>
#include <Datatypes/ContourSet.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/ScalarField.h>
#include <Datatypes/Surface.h>
#include <Datatypes/VectorField.h>

typedef LockingHandle<ContourSet> _dummy1_;
typedef LockingHandle<ScalarField> _dummy2_;
typedef LockingHandle<Surface> _dummy3_;
typedef LockingHandle<VectorField> _dummy4_;
typedef LockingHandle<Mesh> _dummy5_;
typedef LockingHandle<Matrix> _dummy6_;

static void _fn1_(Piostream& p1, MeshHandle& p2)
{
    Pio(p1, p2);
}
static void _fn2_(Piostream& p1, ScalarFieldHandle& p2)
{
    Pio(p1, p2);
}
static void _fn3_(Piostream& p1, ContourSetHandle& p2)
{
    Pio(p1, p2);
}
static void _fn4_(Piostream& p1, SurfaceHandle& p2)
{
    Pio(p1, p2);
}
static void _fn5_(Piostream& p1, VectorFieldHandle& p2)
{
    Pio(p1, p2);
}
static void _fn6_(Piostream& p1, MatrixHandle& p2)
{
    Pio(p1, p2);
}
