
#include <Packages/rtrt/Core/Shadows/ShadowBase.h>
#include <Core/Persistent/Persistent.h>

using namespace rtrt;

// initialize the static member type_id
SCIRun::PersistentTypeID ShadowBase::type_id("ShadowBase", "Persistent", 0);

char * ShadowBase::shadowTypeNames[] = { "No Shadows",
					 "Single Soft Shadow",
					 "Hard Shadows",
					 "Glass Shadows",
					 "Soft Shadows",
					 "Uncached Shadows" };

ShadowBase::ShadowBase()
  : name("unknown")
{
}

ShadowBase::~ShadowBase()
{
}

void ShadowBase::preprocess(Scene*, int&, int&)
{
}

const int SHADOWBASE_VERSION = 1;
void 
ShadowBase::io(SCIRun::Piostream &str)
{
  str.begin_class("ShadowBase", SHADOWBASE_VERSION);
  //  Pio(str, matl);
  //Pio(str, uv);
  str.end_class();
}

namespace SCIRun {
void Pio(Piostream& stream, rtrt::ShadowBase*& obj)
{
  Persistent* pobj=obj;
  stream.io(pobj, rtrt::ShadowBase::type_id);
  if(stream.reading())
    obj=(rtrt::ShadowBase*)pobj;
}
} // end namespace SCIRun
