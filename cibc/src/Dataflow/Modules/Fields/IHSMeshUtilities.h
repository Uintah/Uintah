#include <vector>

#ifndef GTB_POINT3_INCLUDED
#define GTB_POINT3_INCLUDED

GTB_BEGIN_NAMESPACE

template<class T>
class tVector3;

template<class T>
class tPoint3 {
public:
  typedef T value_type;
  tPoint3();
  tPoint3(value_type px, value_type py, value_type pz);

  tPoint3 &operator=(const tPoint3 &p);
  
  value_type x() const;
  value_type y() const;
  value_type z() const;
  
  value_type operator[](unsigned i) const;
  value_type &operator[](unsigned i);  
  
  void add(const tPoint3& p);
  void scalar_scale(value_type s);  

  static tPoint3 centroid(const std::vector<tPoint3> &v);
  
  friend class tVector3<T>;
  
protected:
  value_type _p[3];
};

typedef tPoint3<double> Point3;

//begin inline functions...
template<class T>
inline tPoint3<T>::tPoint3()
{
}

template<class T>
inline tPoint3<T>::tPoint3(value_type px, value_type py, value_type pz)
{
	_p[0] = px;
	_p[1] = py;
	_p[2] = pz;
}

template<class T>
inline tPoint3<T> &tPoint3<T>::operator=(const tPoint3 &p)
{
	_p[0] = p._p[0];
	_p[1] = p._p[1];
	_p[2] = p._p[2];
	return *this;
}

template<class T>
inline typename tPoint3<T>::value_type tPoint3<T>::x() const
{
	return _p[0];
}

template<class T>
inline typename tPoint3<T>::value_type tPoint3<T>::y() const
{
	return _p[1];
}

template<class T>
inline typename tPoint3<T>::value_type tPoint3<T>::z() const
{
	return _p[2];
}

template<class T>
inline typename tPoint3<T>::value_type tPoint3<T>::operator[](unsigned i) const
{
	ASSERT(i < 3);
	return _p[i];
}

template<class T>
inline typename tPoint3<T>::value_type &tPoint3<T>::operator[](unsigned i)
{
	ASSERT(i < 3);
	return _p[i];
}

template<class T>
inline void tPoint3<T>::add(const tPoint3& p)
{
 	_p[0] += p._p[0];
 	_p[1] += p._p[1];
 	_p[2] += p._p[2];
}

template<class T>
inline void tPoint3<T>::scalar_scale(value_type s)
{
 	_p[0] *= s;
 	_p[1] *= s;
 	_p[2] *= s;
}

template<class T>
inline tVector3<T> operator-(const tPoint3<T> &p, const tPoint3<T> &q)
{
	return tVector3<T>(p[0] - q[0], p[1] - q[1], p[2] - q[2]);
}

template<class T>
inline tPoint3<T> tPoint3<T>::centroid(const std::vector<tPoint3> &v)
{
	value_type cx = 0.0;
	value_type cy = 0.0;
	value_type cz = 0.0;

	for (unsigned i = 0; i < v.size(); i++) {
		cx += v[i].x() / v.size();
		cy += v[i].y() / v.size();
		cz += v[i].z() / v.size();
	}
	return tPoint3(cx, cy, cz);
}

GTB_END_NAMESPACE
#endif // GTB_POINT3_INCLUDED


#ifndef GTB_VECTOR3_INCLUDED
#define GTB_VECTOR3_INCLUDED

GTB_BEGIN_NAMESPACE

template<class T>
class tVector3 
{
public:
  typedef T value_type;
  
  tVector3();
  tVector3(value_type vx, value_type vy, value_type vz);
  tVector3 &operator=(const tVector3 &v);
  
  value_type operator[](unsigned i) const;
  tVector3 &operator+=(const tVector3 &v);
  tVector3 &operator/=(value_type a);  
  
  value_type dot(const tVector3 &v) const;
  tVector3 cross(const tVector3 &v) const;
  value_type length() const;
  tVector3 &normalize();
  
protected:
  value_type _v[3];
};

typedef tVector3<double> Vector3;

//inline function definitions....
template<class T>
inline tVector3<T>::tVector3()
{
}

template<class T>
inline tVector3<T>::tVector3(value_type vx, value_type vy, value_type vz)
{
	_v[0] = vx;
	_v[1] = vy;
	_v[2] = vz;
}

template<class T>
inline tVector3<T> &tVector3<T>::operator=(const tVector3 &v)
{
	_v[0] = v._v[0];
	_v[1] = v._v[1];
	_v[2] = v._v[2];
	return *this;
}

template<class T>
inline typename tVector3<T>::value_type tVector3<T>::operator[](unsigned i) const
{
	ASSERT(i < 3);
	return _v[i];
}

template<class T>
inline tVector3<T> &tVector3<T>::operator+=(const tVector3 &v)
{
	_v[0] += v._v[0];
	_v[1] += v._v[1];
	_v[2] += v._v[2];
	return *this;
}

template<class T>
inline tVector3<T> &tVector3<T>::operator/=(value_type a)
{
 	_v[0] /= a;
 	_v[1] /= a;
 	_v[2] /= a;
 	return *this;
}

template<class T>
inline typename tVector3<T>::value_type tVector3<T>::dot(const tVector3 &v) const
{
	return (_v[0] * v._v[0] + _v[1] * v._v[1] + _v[2] * v._v[2]);
}

template<class T>
inline tVector3<T> operator*(T a, const tVector3<T> &v)
{
	return tVector3<T>(a * v[0], a * v[1], a * v[2]);
}

template<class T>
inline tVector3<T> operator*(const tVector3<T> &v, T a)
{
	return a * v;
}

template<class T>
inline tVector3<T> tVector3<T>::cross(const tVector3<T> &v) const
{
	return tVector3<T>((_v[1] * v._v[2]) - (_v[2] * v._v[1]),
                     (_v[2] * v._v[0]) - (_v[0] * v._v[2]),
                     (_v[0] * v._v[1]) - (_v[1] * v._v[0]));
}

template<>
inline double tVector3<double>::length() const
{
	return sqrt(this->dot(*this));
}

template<class T>
inline tVector3<T> &tVector3<T>::normalize()
{
	*this /= length();
	return *this;
}

GTB_END_NAMESPACE
#endif // GTB_VECTOR3_INCLUDED


#ifndef GTB_BOX3_INCLUDED
#define GTB_BOX3_INCLUDED

GTB_BEGIN_NAMESPACE

template<class T>
class tBox3 
{
public:
  typedef T value_type;
  enum { INSIDE, INTERSECT, OUTSIDE };
  
//   enum Face { L, R, D, U, B, F };
  
//   enum Vertex { LDB, LDF, LUB, LUF, RDB, RDF, RUB, RUF };
  
//   enum VertexMask {
//       RIGHT_MASK = 1 << 2,
//       UP_MASK =	1 << 1,
//       FRONT_MASK = 1 << 0
//   };
  
  tBox3();
  tBox3(const tPoint3<T> &min_pt, const tPoint3<T> &max_pt);
  tBox3(value_type x_min, value_type y_min, value_type z_min,
        value_type x_max, value_type y_max, value_type z_max);
  tBox3 &operator=(const tBox3 &b);
  
  value_type x_min() const;
  value_type y_min() const;
  value_type z_min() const;
  
  value_type x_max() const;
  value_type y_max() const;
  value_type z_max() const;
  
  void update( const tPoint3<T>& p );
  void update( double x, double y, double z );
  
  value_type x_length() const;
  value_type y_length() const;
  value_type z_length() const;
  
  tPoint3<T> centroid() const;
  int classify_position(const tBox3 &b) const;
  
  static tBox3 bounding_box(const tPoint3<T> &a, const tPoint3<T> &b, 
                            const tPoint3<T> &c);
  static tBox3 bounding_box(const std::vector<tPoint3<T> > &v);
  static tBox3 make_union( const tBox3 &b1, const tBox3 &b2 );
//   static tBox3 make_union( double b1xmin, double b1ymin, double b1zmin, 
//                            double b1xmax, double b1ymax, double b1zmax,
//                            double b2xmin, double b2ymin, double b2zmin, 
//                            double b2xmax, double b2ymax, double b2zmax );
  
protected:
  bool is_order_correct() const;
  
  tPoint3<T> _min_pt, _max_pt;
  static const unsigned _vertex_indices[6][4];
};

typedef tBox3<double> Box3;

//begin inline functions...
template<class T>
inline bool tBox3<T>::is_order_correct() const
{
	return((_min_pt.x() <= _max_pt.x()) &&
	       (_min_pt.y() <= _max_pt.y()) &&
	       (_min_pt.z() <= _max_pt.z()));
}

template<class T>
inline tBox3<T>::tBox3()
{
}

template<class T>
inline tBox3<T>::tBox3(const tPoint3<T> &min_pt, const tPoint3<T> &max_pt)
        : _min_pt(min_pt),
          _max_pt(max_pt)
{
	ASSERT(is_order_correct());
}

template<class T>
inline tBox3<T>::tBox3(value_type xmin, value_type ymin, value_type zmin,
                       value_type xmax, value_type ymax, value_type zmax)
        : _min_pt(xmin, ymin, zmin),
          _max_pt(xmax, ymax, zmax)
{
	ASSERT(is_order_correct());
}

template<class T>
inline tBox3<T> &tBox3<T>::operator=(const tBox3 &b)
{
  if (&b != this) {
    _min_pt = b._min_pt;
    _max_pt = b._max_pt;
    ASSERT(is_order_correct());
  }
  return *this;
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::x_min() const
{
  return _min_pt.x();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::y_min() const
{
	return _min_pt.y();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::z_min() const
{
  return _min_pt.z();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::x_max() const
{
  return _max_pt.x();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::y_max() const
{
  return _max_pt.y();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::z_max() const
{
  return _max_pt.z();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::x_length() const
{
  return _max_pt.x() - _min_pt.x();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::y_length() const
{
  return _max_pt.y() - _min_pt.y();
}

template<class T>
inline typename tBox3<T>::value_type tBox3<T>::z_length() const
{
  return _max_pt.z() - _min_pt.z();
}

template<class T>
inline tPoint3<T> tBox3<T>::centroid() const
{
  return tPoint3<T>((_min_pt.x() + _max_pt.x()) / 2.0,
                    (_min_pt.y() + _max_pt.y()) / 2.0,
                    (_min_pt.z() + _max_pt.z()) / 2.0);
}

template<class T>
inline tBox3<T> tBox3<T>::bounding_box(const tPoint3<T> &a,
                                       const tPoint3<T> &b,
                                       const tPoint3<T> &c)
{
  return tBox3(std::min(std::min(a[0], b[0]), c[0]),
               std::min(std::min(a[1], b[1]), c[1]),
               std::min(std::min(a[2], b[2]), c[2]),
               std::max(std::max(a[0], b[0]), c[0]),
               std::max(std::max(a[1], b[1]), c[1]),
               std::max(std::max(a[2], b[2]), c[2]));
}

template<class T>
inline void tBox3<T>::update(const tPoint3<T>& p)
{
  for (int i = 0; i < 3; ++i)
  {
    _min_pt[i] = std::min(_min_pt[i], p[i]);
    _max_pt[i] = std::max(_max_pt[i], p[i]);
  }
}

template<class T>
inline void tBox3<T>::update( double x, double y, double z )
{
    _min_pt[0] = std::min(_min_pt[0], x );
    _max_pt[0] = std::max(_max_pt[0], x );
    _min_pt[1] = std::min(_min_pt[1], y );
    _max_pt[1] = std::max(_max_pt[1], y );
    _min_pt[2] = std::min(_min_pt[2], z );
    _max_pt[2] = std::max(_max_pt[2], z );
}

template<class T>
inline tBox3<T> tBox3<T>::bounding_box(const std::vector<tPoint3<T> > &v)
{
	ASSERT(v.size() > 0);
	tPoint3<T> p = v[0];
	value_type min_x = p.x(), min_y = p.y(), min_z = p.z();
	value_type max_x = p.x(), max_y = p.y(), max_z = p.z();
	for (unsigned i = 1; i < v.size(); i++) {
		p = v[i];
		min_x = min(min_x, p.x());
		min_y = min(min_y, p.y());
		min_z = min(min_z, p.z());
		max_x = max(max_x, p.x());
		max_y = max(max_y, p.y());
		max_z = max(max_z, p.z());
	}
	return tBox3(min_x, min_y, min_z,
               max_x, max_y, max_z);
}

template<class T>
inline int tBox3<T>::classify_position(const tBox3 &b) const
{
	if ((_max_pt[0] < b._min_pt[0]) || (_min_pt[0] > b._max_pt[0]) ||
	    (_max_pt[1] < b._min_pt[1]) || (_min_pt[1] > b._max_pt[1]) ||
	    (_max_pt[2] < b._min_pt[2]) || (_min_pt[2] > b._max_pt[2])) {
		return OUTSIDE;
	}
  
	if ((_min_pt[0] <= b._min_pt[0]) && (_max_pt[0] >= b._max_pt[0]) &&
	    (_min_pt[1] <= b._min_pt[1]) && (_max_pt[1] >= b._max_pt[1]) &&
	    (_min_pt[2] <= b._min_pt[2]) && (_max_pt[2] >= b._max_pt[2])) {
		return INSIDE;
  }
  
	return INTERSECT;
}

template<class T>
inline tBox3<T> tBox3<T>::make_union(const tBox3 &b1, const tBox3 &b2)
{
 	return tBox3(min(b1.x_min(), b2.x_min()),
               min(b1.y_min(), b2.y_min()),
               min(b1.z_min(), b2.z_min()),
               max(b1.x_max(), b2.x_max()),
               max(b1.y_max(), b2.y_max()),
               max(b1.z_max(), b2.z_max()));
}

// template<class T>
// inline tBox3<T> tBox3<T>::make_union( double b1xmin, double b1ymin, double b1zmin, 
//                                       double b1xmax, double b1ymax, double b1zmax,
//                                       double b2xmin, double b2ymin, double b2zmin, 
//                                       double b2xmax, double b2ymax, double b2zmax )
// {
// 	return tBox3(min(b1xmin, b2xmin), min(b1ymin, b2ymin), min(b1zmin, b2zmin),
//                max(b1xmax, b2xmax), max(b1ymax, b2ymax), max(b1zmax, b2zmax));
// }

GTB_END_NAMESPACE
#endif // GTB_BOX3_INCLUDED
