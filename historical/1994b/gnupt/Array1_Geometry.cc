
#pragma implementation "Array1.h"

#include <Classlib/Array1.cc>
#include <Classlib/String.h>
#include <Datatypes/TriSurface.h>
#include <Geom/Geom.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>

class GeomObj;

typedef Array1<Point> _dummy1_;
typedef Array1<Vector> _dummy2_;
typedef Array1<Array1<Point> > _dummy3_;

static void _dummy4_(Piostream& p1, _dummy3_& p2)
{
    Pio(p1, p2);
}

typedef Array1<TSElement*> _dummy10_;
typedef Array1<GeomObj*> _dummy11_;
typedef Array1<MaterialHandle> _dummy12_;

static void _fn1_(Piostream& p1, Array1<TSElement*>& p2)
{
    Pio(p1, p2);
}

static void _fn1_(Piostream& p1, Array1<Vector>& p2)
{
    Pio(p1, p2);
}
