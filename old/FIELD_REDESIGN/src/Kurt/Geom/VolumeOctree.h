#ifndef VOLUME_OCTREE_H
#define VOLUME_OCTREE_H
#include <SCICore/Geometry/Point.h>

namespace Kurt {
namespace GeomSpace {

using namespace SCICore::Geometry;
#include <SCICore/Geometry/Point.h>

using SCICore::Geometry::Point;

template<class T>
class VolumeOctree {
public:
  enum nodeType {LEAF, PARENT};

  VolumeOctree(Point min, Point max, T stored, int nodeId,
	       VolumeOctree::nodeType t);
  ~VolumeOctree();
  void SetChild(int i, VolumeOctree<T>* n);
  const VolumeOctree<T>* child(int i) const;
  nodeType getType() const {return t;}
  int getId() const{return id;}
  T operator()() const;
  Point getMin() const { return min; }
  Point getMax() const { return max; } 
private:
  T stored;
  int level;
  nodeType t;
  VolumeOctree <T> **children;
  Point min, max;
  int id;
};

} // end namespace GeomSpace
} // end namespace Kurt
#endif
