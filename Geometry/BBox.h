#ifndef Geometry_BBox_h
#define Geometry_BBox_h 1

#include <Geometry/Point.h>
#include <Geometry/Plane.h>
#include <Tester/RigorousTest.h>
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
    //////////
    //Default Constructor
    BBox();
    //////////
    //Destructor
    ~BBox();
    //////////
    //Copy Constructor
    BBox(const BBox&);
    //////////
    //Is the BBox valid?
    inline int valid() const {return have_some;}

    void reset();
    void extend(const Point& p);
    void extend(const Point& p, double radius);
    void extend(const BBox& b);
    //////////
    //
    void extend_cyl(const Point& cen, const Vector& normal, double r);
    //////////
    //Return the center point of the BBox
    Point center() const;
    //////////
    //Return the length of the largest edge
    double longest_edge();
    //////////
    //Move the BBox in space
    void translate(const Vector &v);
    //////////
    //Change the dimensions of the box by a scale, and translate the box
    void scale(double s, const Vector &o);
    //////////
    //Return the minimum of the BBox
    Point min() const;
    //////////
    //Returns the maximum of the BBox
    Point max() const;
    //////////
    //Returns the diagonal vector of the BBox
    Vector diagonal() const;  
    //////////
    //Is a point within the bounds of a BBox?
    inline int inside(const Point &p) const {return (have_some && p.x()>=cmin.x() && p.y()>=cmin.y() && p.z()>=cmin.z() && p.x()<=cmax.x() && p.y()<=cmax.y() && p.z()<=cmax.z());}

    //Rigorous Tests
    static void test_rigorous(RigorousTest* __test);
  
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
