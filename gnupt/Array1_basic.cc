
#pragma implementation "Array1.h"

#include <Classlib/Array1.cc>
#include <Classlib/String.h>
#include <Geometry/Point.h>
#include <Geometry/Vector.h>
#include <Mesh.h>

class Connection;
class Element;
class GeomItem;
class IPort;
class Module;
class MUI_widget;
class Node;
class OPort;
class Roe;
class SLine;
class SoundMixer_PortInfo;
class SSLine;
class SSurf;
class TCLvar;

typedef Array1<Connection*> _dummy1_;
typedef Array1<GeomItem*> _dummy2_;
typedef Array1<IPort*> _dummy4_;
typedef Array1<Module*> _dummy5_;
typedef Array1<OPort*> _dummy6_;
typedef Array1<Roe*> _dummy7_;
typedef Array1<SLine*> _dummy8_;
typedef Array1<SoundMixer_PortInfo*> _dummy9_;

typedef Array1<int> _dummy11_;
typedef Array1<MUI_widget*> _dummy12_;
typedef Array1<clString> _dummy13_;
typedef Array1<Node*> _dummy14_;
typedef Array1<Element*> _dummy15_;
typedef Array1<double> _dummy16_;
typedef Array1<SSLine*> _dummy17_;
typedef Array1<SSurf*> _dummy18_;
typedef Array1<TCLvar*> _dummy19_;

static void _fn1_(Piostream& p1, Array1<Node*>& p2)
{
    Pio(p1, p2);
}

static void _fn2_(Piostream& p1, Array1<Element*>& p2)
{
    Pio(p1, p2);
}

static void _fn3_(Piostream& p1, Array1<double>& p2)
{
    Pio(p1, p2);
}

