
#include <Packages/rtrt/Core/Shadows/ShadowBase.h>
using namespace rtrt;

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

