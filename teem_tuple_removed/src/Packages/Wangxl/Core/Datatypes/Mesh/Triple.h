#ifndef SCI_Wangxl_Datatypes_Mesh_Triple_h
#define SCI_Wangxl_Datatypes_Mesh_Triple_h

namespace Wangxl {

using namespace SCIRun;

template <class T1, class T2, class T3>
struct triple 
{
  T1 first;
  T2 second;
  T3 third;

  triple() {}

  triple(const T1& a, const T2& b, const T3& c)
    : first(a), second(b), third(c) 
    {}
};

template <class T1, class T2, class T3>
inline 
triple<T1, T2, T3> make_triple(const T1& x, const T2& y, const T3& z)
{
  return triple<T1, T2, T3>(x, y, z);
}

template <class T1, class T2, class T3>
inline bool operator==(const triple<T1, T2, T3>& x,
		       const triple<T1, T2, T3>& y) 
{ 
  return ( (x.first == y.first) && 
	   (x.second == y.second) && 
	   (x.third == y.third) ); 
}

template <class T1, class T2, class T3>
inline
bool operator<(const triple<T1, T2, T3>& x,
	       const triple<T1, T2, T3>& y)
{ 
  return ( x.first < y.first || 
	   ( (x.first == y.first) && (x.second < y.second) ) ||
	   ( (x.first == y.first) && (x.second == y.second) && 
	                             (x.third < y.third) ) );
}

}

#endif
