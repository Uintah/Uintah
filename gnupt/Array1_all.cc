
#pragma implementation "Array1.h"

#include <Classlib/Array1.cc>
#include <Classlib/String.h>
#include <Constraints/BaseConstraint.h>
#include <Constraints/ConstraintSolver.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geometry/BSphere.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Datatypes/ContourSet.h>
#include <Datatypes/ContourSetPort.h>
#include <Datatypes/Mesh.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/TriSurface.h>

class Connection;
class DebugSwitch;
class GeomGroup;
class GeomObj;
class GeomPick;
class GeomPolyline;
class GeomSwitch;
class GeomTriStrip;
class GeomTube;
class GeomVertex;
class IPort;
class Light;
class Module;
class Node;
class OPort;
class PathPoint;
class Roe;
class SLine;
class SLSource;
class SLTracer;
class SRibbon;
class SoundMixer_PortInfo;
class SSLine;
class SSurf;
class TCLvar;
class TCLvarintp;

typedef Array1<Connection*> _dummy1_;
typedef Array1<IPort*> _dummy4_;
typedef Array1<Module*> _dummy5_;
typedef Array1<OPort*> _dummy6_;
typedef Array1<Roe*> _dummy7_;
typedef Array1<SLine*> _dummy8_;
typedef Array1<SoundMixer_PortInfo*> _dummy9_;

typedef Array1<int> _dummy11_;
typedef Array1<clString> _dummy13_;
typedef Array1<Node*> _dummy14_;
typedef Array1<Element*> _dummy15_;
typedef Array1<double> _dummy16_;
typedef Array1<SSLine*> _dummy17_;
typedef Array1<SSurf*> _dummy18_;
typedef Array1<TCLvar*> _dummy19_;

typedef Array1<unsigned int> _dummy20_;
typedef Array1<BaseConstraint*> _dummy22_;
typedef Array1<VPriority> _dummy24_;
typedef Array1<Light*> _dummy25_;
typedef Array1<ContourSetIPort*> _dummy26_;
typedef Array1<ContourSetHandle> _dummy27_;
typedef Array1<TCLvarintp*> _dummy28_;

typedef Array1<Point> _dummy30_;
typedef Array1<Vector> _dummy31_;
typedef Array1<Array1<Point> > _dummy32_;
typedef Array1<SRibbon*> _dummy33_;

static void _dummy33_(Piostream& p1, _dummy32_& p2)
{
    Pio(p1, p2);
}

typedef Array1<TSElement*> _dummy34_;
typedef Array1<GeomObj*> _dummy35_;
typedef Array1<MaterialHandle> _dummy36_;
typedef Array1<SurfaceIPort*> _dummy37_;
typedef Array1<SurfaceHandle> _dummy38_;
typedef Array1<DebugSwitch*> _dummy39_;

typedef Array1<GeomGroup::ITree*> _dummy40_;
typedef Array1<BSphere> _dummy41_;
typedef Array1<GeomVertex*> _dummy42_;
typedef Array1<SLTracer*> _dummy43_;
typedef Array1<GeomGroup*> _dummy44_;
typedef Array1<GeomPolyline*> _dummy45_;
typedef Array1<GeomSwitch*> _dummy46_;
typedef Array1<GeomTube*> _dummy47_;
typedef Array1<GeomTriStrip*> _dummy48_;
typedef Array1<PathPoint*> _dummy49_;

typedef Array1<SLSource*> _dummy50_;
typedef Array1<GeomPick*> _dummy51_;
typedef Array1<StackItem> _dummy52_;
typedef Array1<long> _dummy53_;
typedef Array1<MeshHandle> _dummy54_;
typedef Array1<NodeVersion1> _dummy55_;

static void _fn1_(Piostream& p1, Array1<TSElement*>& p2)
{
    Pio(p1, p2);
}

static void _fn2_(Piostream& p1, Array1<Vector>& p2)
{
    Pio(p1, p2);
}
static void _fn4_(Piostream& p1, Array1<Element*>& p2)
{
    Pio(p1, p2);
}

static void _fn5_(Piostream& p1, Array1<double>& p2)
{
    Pio(p1, p2);
}

static void _fn6_(Piostream& p1, Array1<MaterialHandle>& p2)
{
    Pio(p1, p2);
}

static void _fn7_(Piostream& p1, Array1<NodeHandle>& p2)
{
    Pio(p1, p2);
}

static void _fn8_(Piostream& p1, Array1<NodeVersion1>& p2)
{
    Pio(p1, p2);
}

static void _fn9_(Piostream& p1, Array1<MeshHandle>& p2)
{
    Pio(p1, p2);
}
