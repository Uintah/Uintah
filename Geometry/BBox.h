
#ifndef Geometry_BBox_h
#define Geometry_BBox_h 1

#include <Geometry/Point.h>
#include <Geometry/Plane.h>
class Vector;
class Piostream;

class BBoxSide;
class BBox;

class BBoxSide
{
  Plane pl;
  Point pts[2][3];

public:  
  Vector perimeter[2][3];
  BBoxSide *next;
  
  BBoxSide( );
  BBoxSide( Point a, Point b, Point c, Point d );
  void ChangeBBoxSide( Point a, Point b, Point c, Point d );

  int BBoxSide::Intersect( Point s, Vector v, Point& hit );
};



class BBox {
protected:
friend void Pio(Piostream &, BBox& );
    int have_some;
    Point cmin;
    Point cmax;
    BBoxSide sides[6];
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
    Point corner(int) const;

    // sets up the bbox sides for intersection calculations

    void SetupSides( );
    
    // finds up to 2 points of intersection of a ray and a bbox

    int Intersect( Point s, Vector v, Point& hitNear, Point& hitFar );
};

#endif
