
#include <Packages/rtrt/Core/Shadows/NoShadows.h>
using namespace rtrt;


NoShadows::NoShadows()
{
}

NoShadows::~NoShadows()
{
}

bool NoShadows::lit(const Point& hitpos, Light* light,
		    const Vector& light_dir, double dist, Color& atten,
		    int depth, Context* cx)
{
  return true;
}

