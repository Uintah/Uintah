
#pragma implementation "Array1.h"

#include <Classlib/Array1.cc>
#include <Classlib/String.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Surface.h>

typedef Array1<Point> _dummy1_;
typedef Array1<Vector> _dummy2_;
typedef Array1<Array1<Point> > _dummy3_;
static void _dummy4_()
{
    Array1<Array1<Point> >* xx;
    Piostream* yy;
    Pio(*yy, *xx);
}

typedef Array1<TSElement*> _dummy10_;
typedef Array1<GeomObj*> _dummy11_;

static void _fn1_(Piostream& p1, Array1<TSElement*>& p2)
{
    Pio(p1, p2);
}

static void _fn1_(Piostream& p1, Array1<Vector>& p2)
{
    Pio(p1, p2);
}
