

#pragma implementation "ITC.h"

#include <Datatypes/GeometryComm.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/ContourSetPort.h>
#include <Datatypes/Matrix.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/SurfacePort.h>
#include <Datatypes/VectorFieldPort.h>

class MessageBase;
class SoundComm;

typedef Mailbox<MessageBase*> _dummy1_;
typedef Mailbox<GeomReply> _dummy2_;
typedef Mailbox<SoundComm*> _dummy3_;
typedef Mailbox<SimplePortComm<ContourSetHandle>*> _dummy4_;
typedef Mailbox<SimplePortComm<ScalarFieldHandle>*> _dummy5_;
typedef Mailbox<SimplePortComm<SurfaceHandle>*> _dummy6_;
typedef Mailbox<SimplePortComm<VectorFieldHandle>*> _dummy7_;
typedef Mailbox<SimplePortComm<MeshHandle>*> _dummy8_;
typedef Mailbox<SimplePortComm<MatrixHandle>*> _dummy9_;
typedef Mailbox<SimplePortComm<ColormapHandle>*> _dummy10_;
typedef Mailbox<SimplePortComm<ColumnMatrixHandle>*> _dummy11_;
