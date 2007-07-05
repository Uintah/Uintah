// Bounding box that you can add points to.
// by David K. McAllister, 1998.

#ifndef _bbox_h
#define _bbox_h

#include <Packages/Remote/Tools/Math/Vector.h>

namespace Remote {
class BBox
{
	bool valid;

public:
	Vector MinV, MaxV;

	inline BBox() {valid = false;}
	inline void Reset() {valid = false;}
	inline double MaxDim() const
	{
		return Max(MaxV.x - MinV.x, MaxV.y - MinV.y, MaxV.z - MinV.z);
	}

	inline Vector Center() const
  	{
	  return MinV + (MaxV - MinV) * 0.5;
	}

	// Returns true if the point is in the box.
	inline bool Inside(const Vector &P) const
	{
	  return !((P.x < MinV.x || P.y < MinV.y || P.z < MinV.z || 
		  P.x > MaxV.x || P.y > MaxV.y || P.z > MaxV.z));
	}

	// Returns true if any of the sphere is in the box.
	// XXX For speed we will return true if the P+-r box intersects the bbox.
	inline bool SphereIntersect(const Vector &P, const double r) const
	{
	  return (P.x+r < MinV.x || P.y+r < MinV.y || P.z+r < MinV.z || 
		  P.x-r > MaxV.x || P.y-r > MaxV.y || P.z-r > MaxV.z);
	}

	inline BBox& operator+=(const Vector &v)
	{
		if(valid)
		{
			MinV.x = Min(MinV.x, v.x);
			MinV.y = Min(MinV.y, v.y);
			MinV.z = Min(MinV.z, v.z);

			MaxV.x = Max(MaxV.x, v.x);
			MaxV.y = Max(MaxV.y, v.y);
			MaxV.z = Max(MaxV.z, v.z);
		}
		else
		{
			valid = true;
			MinV = v;
			MaxV = v;
		}

		return *this;
	}

	inline BBox& operator+=(const BBox &b)
	{
		if(valid)
		{
			MinV.x = Min(MinV.x, b.MinV.x);
			MinV.y = Min(MinV.y, b.MinV.y);
			MinV.z = Min(MinV.z, b.MinV.z);

			MaxV.x = Max(MaxV.x, b.MaxV.x);
			MaxV.y = Max(MaxV.y, b.MaxV.y);
			MaxV.z = Max(MaxV.z, b.MaxV.z);
		}
		else
		{
			valid = true;
			MinV = b.MinV;
			MaxV = b.MaxV;
		}

		return *this;
	}
};

inline ostream& operator<<(ostream& os, const BBox& b)
{
   os << b.MinV.print() << b.MaxV.print();
   return os;
}

inline istream& operator>>(istream& is, BBox& b)
{
  is >> b.MinV >> b.MaxV;
  return is;
}

} // End namespace Remote


#endif
