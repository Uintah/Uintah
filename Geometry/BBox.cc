
#include <Geometry/BBox.h>
#include <Geometry/Vector.h>
#include <Classlib/Assert.h>
#include <Classlib/Exceptions.h>
#include <Math/MinMax.h>
#include <Classlib/Persistent.h>

BBox::BBox()
{
    have_some=0;
}

BBox::BBox(const BBox& copy)
: have_some(copy.have_some), cmin(copy.cmin), cmax(copy.cmax) 
{
}
    
BBox::~BBox()
{
}

void BBox::reset()
{
    have_some=0;
}

void BBox::extend(const Point& p)
{
    if(have_some){
	cmin=Min(p, cmin);
	cmax=Max(p, cmax);
    } else {
	cmin=p;
	cmax=p;
	have_some=1;
    }
}

void BBox::extend(const Point& p, double radius)
{
    Vector r(radius,radius,radius);
    if(have_some){
	cmin=Min(p-r, cmin);
	cmax=Max(p+r, cmax);
    } else {
	cmin=p-r;
	cmax=p+r;
	have_some=1;
    }
}

void BBox::extend(const BBox& b)
{
    extend(b.min());
    extend(b.max());
}

void BBox::extend_cyl(const Point& cen, const Vector& normal, double r)
{
    Vector n(normal.normal());
    double x=Sqrt(1-n.x())*r;
    double y=Sqrt(1-n.y())*r;
    double z=Sqrt(1-n.z())*r;
    extend(cen+Vector(x,y,z));
    extend(cen-Vector(x,y,z));
}

Point BBox::center()
{
    ASSERT(have_some);
    return Interpolate(cmin, cmax, 0.5);
}

void BBox::translate(const Vector &v) {
    cmin+=v;
    cmax+=v;
}

void BBox::scale(double s, const Vector&o) {
    cmin-=o;
    cmax-=o;
    cmin*=s;
    cmax*=s;
    cmin+=o;
    cmax+=o;
}

double BBox::longest_edge()
{
    ASSERT(have_some);
    Vector diagonal(cmax-cmin);
    return Max(diagonal.x(), diagonal.y(), diagonal.z());
}

Point BBox::min() const
{
    ASSERT(have_some);
    return cmin;
}

Point BBox::max() const
{
    ASSERT(have_some);
    return cmax;
}

Vector BBox::diagonal() const
{
    ASSERT(have_some);
    return cmax-cmin;
}

Point BBox::corner(int i)
{
    ASSERT(have_some);
    switch(i){
    case 0:
	return Point(cmin.x(), cmin.y(), cmin.z());
    case 1:
	return Point(cmin.x(), cmin.y(), cmax.z());
    case 2:
	return Point(cmin.x(), cmax.y(), cmin.z());
    case 3:
	return Point(cmin.x(), cmax.y(), cmax.z());
    case 4:
	return Point(cmax.x(), cmin.y(), cmin.z());
    case 5:
	return Point(cmax.x(), cmin.y(), cmax.z());
    case 6:
	return Point(cmax.x(), cmax.y(), cmin.z());
    case 7:
	return Point(cmax.x(), cmax.y(), cmax.z());
    default:
	EXCEPTION(General("Bad corner requested from BBox"));
    }
    return Point(0,0,0);
}

void Pio(Piostream &stream, BBox& box) {
    stream.begin_cheap_delim();
    Pio(stream, box.have_some);
    if (box.have_some) {
	Pio(stream, box.cmin);
	Pio(stream, box.cmax);
    }
    stream.end_cheap_delim();
}
