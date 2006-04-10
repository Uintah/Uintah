#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>
using namespace std;

class Point
{
public:
  Point()
    : d_id(-999999) {}
  
  Point(int id, int x, int y, int z)
    : d_id(id)
  { d_vals[0] = x; d_vals[1] = y; d_vals[2] = z; }

  /*
  Point(const Point& p)
    : d_id(p.d_id)
  {
    d_vals[0] = p.d_vals[0]; d_vals[1] = p.d_vals[1]; d_vals[2] = p.d_vals[2];
    }*/
    
  Point& operator=(const Point& p)
  {
    d_id = p.d_id;
    d_vals[0] = p.d_vals[0]; d_vals[1] = p.d_vals[1]; d_vals[2] = p.d_vals[2];
    return *this;
  }

  int distanceSquared(const Point& p2) const
  { return (d_vals[0] - p2.d_vals[0]) * (d_vals[0] - p2.d_vals[0]) +
      (d_vals[1] - p2.d_vals[1]) * (d_vals[1] - p2.d_vals[1]) +
      (d_vals[2] - p2.d_vals[2]) * (d_vals[2] - p2.d_vals[2]); }
  
  int distanceL1(const Point& p2) const
  { return abs(d_vals[0] - p2.d_vals[0]) + abs(d_vals[1] - p2.d_vals[1]) +
      abs(d_vals[2] - p2.d_vals[2]); }
    
  int getId() const
  { return d_id; }
  
  int operator[](int i) const
  { return d_vals[i]; }

  int& operator[](int i)
  { return d_vals[i]; }
private:
  int d_id;
  int d_vals[3];
};
