
#pragma implementation "Array3.h"

#include <Classlib/Array3.cc>
#include <Classlib/Array1.h>
#include <Classlib/Persistent.h>
#include <Geometry/Vector.h>

typedef Array3<double> _dummy1_;
typedef Array3<Vector> _dummy2_;
typedef Array3<Array1<int>*> _dummy3_;

static void _dummy3_(Piostream& p1, Array3<double>& p2)
{
    Pio(p1, p2);
}

static void _dummy4_(Piostream& p1, Array3<Vector>& p2)
{
    Pio(p1, p2);
}

