

#pragma implementation "ITC.h"

#include <Multitask/ITC.cc>
#include <GeometryPort.h>
#include <ContourSetPort.h>
#include <Matrix.h>
#include <MeshPort.h>
#include <ScalarFieldPort.h>
#include <SurfacePort.h>
#include <VectorFieldPort.h>

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
