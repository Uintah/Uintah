#ifndef Geometry_BBox_h
#define Geometry_BBox_h 1

#include <Geometry/Point.h>
#include <Geometry/Plane.h>
class Vector;
class Piostream;

class BBox;

#define EEpsilon  1e-13

class BBox {
protected:
friend void Pio(Piostream &, BBox& );
    int have_some;
    Point cmin;
    Point cmax;
    Point bcmin, bcmax;
    Point extracmin;
    int inbx, inby, inbz;
public:
    BBox();
    ~BBox();
    BBox(const BBox&);
    inline int valid() const {return have_some;}
    void reset();
    void extend(const Point& p);
    void extend(const Point& p, double radius);
    void extend(const BBox& b);
    void extend_cyl(const Point& cen, const Vector& normal, double r);
    Point center() const;
    double longest_edge();
    void translate(const Vector &v);
    void scale(double s, const Vector &o);
    Point min() const;
    Point max() const;
    Vector diagonal() const;

    // prepares for intersection by assigning the closest bbox corner
    // to extracmin and initializing an epsilon bigger bbox
    
    void PrepareIntersect( const Point& e );
    
    // returns true if the ray hit the bbox and returns the hit point
    // in hitNear

    int Intersect( const Point& e, const Vector& v, Point& hitNear );

    // given a t distance, assigns the hit point.  returns true
    // if the hit point lies on a bbox face

    int TestTx( const Point& e, const Vector& v, double tx, Point& hitNear );
    int TestTy( const Point& e, const Vector& v, double ty, Point& hitNear );
    int TestTz( const Point& e, const Vector& v, double tz, Point& hitNear );
};

#endif
