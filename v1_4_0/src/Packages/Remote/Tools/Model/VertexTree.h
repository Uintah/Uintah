#ifndef _vertex_tree_h
#define _vertex_tree_h

#include <Packages/Remote/Tools/Util/KDTree.h>

namespace Remote {
struct KDVertex
{
  Vertex *Vert;

  inline KDVertex() {Vert = NULL;}

  inline KDVertex(Vertex *Ver) : Vert(Ver) {}

  friend inline bool lessX(const KDVertex &a, const KDVertex &b)
  {
    return a.Vert->V.x < b.Vert->V.x;
  }

  friend inline bool lessY(const KDVertex &a, const KDVertex &b)
  {
    return a.Vert->V.y < b.Vert->V.y;
  }

  friend inline bool lessZ(const KDVertex &a, const KDVertex &b)
  {
    return a.Vert->V.z < b.Vert->V.z;
  }

  // These three are for breaking ties in the KDTree.
  friend inline bool lessFX(const KDVertex &a, const KDVertex &b)
  {
    if(a.Vert->V.x < b.Vert->V.x) return true;
    else if(a.Vert->V.x > b.Vert->V.x) return false;
    else if(a.Vert->V.y < b.Vert->V.y) return true;
    else if(a.Vert->V.y > b.Vert->V.y) return false;
    else return a.Vert->V.z < b.Vert->V.z;
  }

  friend inline bool lessFY(const KDVertex &a, const KDVertex &b)
  {
    if(a.Vert->V.y < b.Vert->V.y) return true;
    else if(a.Vert->V.y > b.Vert->V.y) return false;
    else if(a.Vert->V.z < b.Vert->V.z) return true;
    else if(a.Vert->V.z > b.Vert->V.z) return false;
    else return a.Vert->V.x < b.Vert->V.x;
  }

  friend inline bool lessFZ(const KDVertex &a, const KDVertex &b)
  {
    if(a.Vert->V.z < b.Vert->V.z) return true;
    else if(a.Vert->V.z > b.Vert->V.z) return false;
    else if(a.Vert->V.x < b.Vert->V.x) return true;
    else if(a.Vert->V.x > b.Vert->V.x) return false;
    else return a.Vert->V.y < b.Vert->V.y;
  }

  inline bool operator==(const KDVertex &a) const
  {
    return Vert->V == a.Vert->V;
  }

  inline Vector &vector() const
  {
    return Vert->V;
  }
};

} // End namespace Remote


#endif
