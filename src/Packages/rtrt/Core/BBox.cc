
#include "BBox.h"

using namespace rtrt;

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

