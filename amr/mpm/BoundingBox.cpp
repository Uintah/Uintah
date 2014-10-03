#include"BoundingBox.h"
BoundingBox::BoundingBox(const Vec2D& a, const Vec2D& b):
    from(a), to(b){};

BoundingBox::BoundingBox(const BoundingBox& b)
{
    from = b.GetFrom();
    to = b.GetTo();
}
const BoundingBox& BoundingBox::operator=(const BoundingBox& b)
{
    from = b.GetFrom();
    to = b.GetTo();
}

bool BoundingBox::IsValid()
{
    if(to[x1] >= from[x1] && to[x2] >= from[x2])
        return true;
    else
        return false;
}

bool BoundingBox::Contains(const Vec2D& v) const
{
    //debug code
    /*double tmp;
    if(v[x1] >= from[x1])
        tmp = 1;
    tmp = abs(from[x1] - v[x1]);
    if(v[x1] <= to[x1])
        tmp = 2;
    tmp = abs(to[x1] - v[x1]);
    if(v[x2] >= from[x2])
        tmp = 3;
    tmp = abs(from[x2] - v[x2]);
    if(v[x2] <= to[x2])
        tmp = 4;
    tmp = abs(to[x2] - v[x2]);*/
    if(v[x1] >= from[x1] && v[x1] <= to[x1] && v[x2] >= from[x2] && v[x2] <= to[x2])
        return true;
    else
        return false;
}

bool BoundingBox::Contains(const BoundingBox& b) const
{
    if(Contains(b.GetFrom()) && Contains(b.GetTo()))
        return true;
    else
        return false;
}

bool BoundingBox::IntersectsWith(const BoundingBox& b)
{
    //checking if this boundaing box contains any of the vertices
    //of the given bounding box
    Vec2D bl = b.GetFrom();
    Vec2D tr = b.GetTo();
    Vec2D br = {tr[x1], bl[x2]};
    Vec2D tl = {bl[x1], tr[x2]};
    if(Contains(bl) || Contains(br) || Contains(tr) || Contains(tl))
        return true;
    else
        return false;
}

bool BoundingBox::StrictlyContains(const Vec2D& v)
{
    if(v[x1] > from[x1] && v[x1] < to[x1] && v[x2] > from[x2] && v[x2] < to[x2])
        return true;
    else
        return false;
}

ostream& BoundingBox::operator<<(std::ostream& os)
{
    std::ios state(NULL);
    state.copyfmt(os);
    os.precision(3);
    os << fixed;

    os << "[" << from << "|" << to << "]";

    os.copyfmt(state);
    return os;
}
