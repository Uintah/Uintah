
#include <Packages/rtrt/Core/BBox.h>

using namespace rtrt;
using namespace SCIRun;

BBox::BBox(const BBox& copy)
    : cmin(copy.cmin), cmax(copy.cmax), have_some(copy.have_some)
{
}

void BBox::reset() {
    have_some=false;
}

Point BBox::center() const {
    return cmin+(cmax-cmin)*0.5;
}

