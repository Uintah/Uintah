#ifndef __KDTREE_H
#define __KDTREE_H

#include <vector>
#include <queue>
#include <Dataflow/Modules/Fields/IHSMeshUtilities.h>

/*
 * Author: Shachar Fleishman, shacharf@math.tau.ac.il
 * Description:
 *
 * A (hopefully) stand-alone 3D kd-tree class
 * Assumptions:
 *   1. uses the algebra3 vector objects
 *   2. Objects are points.
 *   3. There is a function GetPoint(object) that returns a reference
 *      to a tPoint3<REAL>. (Assumed to be cheep).
 *      replace by template argument
 *
 * Supported features:
 *  - Insert objects. (does not update the tree)
 *    NOTE: the implementation is memory efficient, inserting
 *          one object at a time is slow, it is better
 *          to use the insert(f,l) function
 *
 *  - MakeTree - Computes the tree.
 *    Insert, MakeTree can be interchanged, i.e.
 *    Insert, MakeTree loop.
 *  - user-defined minimal # of objects/cell
 *  - Tranverse objects in the neighborhood of a point to a given radius / number.
 *  - Traversal methods:
 *    1. by giving a function that will be called with every object
 *       that is in the neighborhood
 *    2. init / getnext mechanism
 *
 * Future:
 *  - Ray trace through tree.
 *  - Clean up traversal and insert
 *  - Make fast traversal non-ordered for points in radius
 *  - Can I make quick epsilon accuracy traversal?
 *
 *  - traverse search with distance^2!!!
 *  - Remove the owner = false in traversel, check scalable???
 *  - remove the points color from render!!!
 */

GTB_BEGIN_NAMESPACE

//  * A priority queue that allows the user to insert many
//  * objects before generating the pq.
template<
    class T,
    class Cont = std::vector<T>,
    class Pred = std::less<typename Cont::value_type> >
class fast_pq
{
public:
  typedef typename Cont::value_type      value_type;
  typedef typename Cont::size_type       size_type;
  typedef          Cont                  container_type;
  
  typedef typename Cont::reference       reference;
  typedef typename Cont::const_reference const_reference;
protected:
  Cont c;
  Pred _comp;
public:
  fast_pq() : c() {c.reserve(100);}
  explicit fast_pq(const Pred& __x) :  c(), _comp(__x) {c.reserve(100);}
  fast_pq(const Pred& __x, const Cont& __s) 
          : c(__s), _comp(__x)
      { 
        make_heap(c.begin(), c.end(), _comp); 
      }
  
    // standard priority queue routines
  bool empty() const { return c.empty(); }
  size_type size() const { return c.size(); }
  const_reference top() const { return c.front(); }
  
  void push(const value_type& __x) 
      {
          c.push_back(__x); 
          std::push_heap(c.begin(), c.end(), _comp);
      }
  void pop() {
      std::pop_heap(c.begin(), c.end(), _comp);
      c.pop_back();
  }
  
    // My extentions
  template <class _InputIterator>
  void insert( _InputIterator __first, _InputIterator __last)
      {
          c.insert(c.end(), __first, __last);
          std::make_heap(c.begin(), c.end(), _comp);
      }  
  
    // insert an element without remaking the heap
    // it is the users responsiblity to remake the heap
    //
    // BUG: should have added an assert on top(), pop()
    // in case the user did not call make heap
  void push_only(const T& x)
      {
          c.push_back(x);
      }
  
  void remake_heap()
      {
          std::make_heap(c.begin(), c.end(), _comp);
      }
  
}; // fast_pq

// kd tree that allows overlapping bounding boxes - each object only exists in the tree in one place

// DISTBOXCLASS	must provide two typenames and two functions:
// typename DISTBOXCLASS::Box3
// typename DISTBOXCLASS::Point3
// DISTBOXCLASS::bounding_box(const OBJECT &) const;
// DISTBOXCLASS::distance(const OBJECT &, const typename DISTBOXCLASS::Point3 &) const;

template <typename OBJECT, typename DISTBOXCLASS>
class BoxKDTree {
  
public:
  
#define BOXKDTREE_MAXLEAFSIZE	10
  
	BoxKDTree() {
		children[0] = children[1] = NULL;
	}
	~BoxKDTree() {
		if (children[0]) delete children[0];
		if (children[1]) delete children[1];
	}
  
	BoxKDTree(const std::vector<OBJECT> &iobjects, const DISTBOXCLASS &boxclass, int axis=0) {
    
		children[0] = children[1] = NULL;
    
		ReBuild(iobjects, boxclass, axis);
	}
  
	void ReBuild(const std::vector<OBJECT> &iobjects, const DISTBOXCLASS &boxclass, int axis=0) {
    
		if (children[0]) delete children[0];
		if (children[1]) delete children[1];
		children[0] = children[1] = NULL;
    
		objects = iobjects;
    
      // make the bounding box
    bbox = boxclass.bounding_box(objects[0]);
		for (unsigned i=1; i<objects.size(); i++) {
			bbox = DISTBOXCLASS::Box3::make_union(bbox, boxclass.bounding_box(objects[i]));
		}
    
		Split(boxclass, axis);
	}
  
    // get all the objects who's bounding boxes intersect the given box
	void GetIntersectedBoxes(const DISTBOXCLASS &boxclass, const typename DISTBOXCLASS::Box3 &ibox, std::vector<OBJECT> &intersected) const {
    
      // check our bounding box
		if (ibox.classify_position(bbox) == DISTBOXCLASS::Box3::OUTSIDE) {
			return;
		}
    
      // check any leaf objects
		for (unsigned i=0; i<objects.size(); i++) {
			typename DISTBOXCLASS::Box3 obox = boxclass.bounding_box(objects[i]);
      
			if (ibox.classify_position(obox) != DISTBOXCLASS::Box3::OUTSIDE) {
				intersected.push_back(objects[i]);
			}
		}
    
      // try going into the children
		if (children[0])	children[0]->GetIntersectedBoxes(boxclass, ibox, intersected);
		if (children[1])	children[1]->GetIntersectedBoxes(boxclass, ibox, intersected);
	}
  
	void Insert(const DISTBOXCLASS &boxclass, const OBJECT &o, int axis=0) {
    
		typename DISTBOXCLASS::Box3 obox = boxclass.bounding_box(o);
    
      // figure out which side we want to put it in
		int addside = -1;
    
		if (children[0] && children[1]) {
        // see which insertion would result in a smaller bounding box overlap
      
			typename DISTBOXCLASS::Box3 c0e = DISTBOXCLASS::Box3::make_union(children[0]->bbox, obox);
			typename DISTBOXCLASS::Box3 c1e = DISTBOXCLASS::Box3::make_union(children[1]->bbox, obox);
      
			bool intersect0 = c0e.classify_position(children[1]->bbox) != DISTBOXCLASS::Box3::OUTSIDE;
			bool intersect1 = c1e.classify_position(children[0]->bbox) != DISTBOXCLASS::Box3::OUTSIDE;
      
			if (intersect0 && !intersect1) {
				addside = 1;
			} else if (!intersect0 && intersect1) {
				addside = 0;
			} else if (intersect0 && intersect1) {
          // figure out which way causes the smallest overlap
        
        typename DISTBOXCLASS::Point3
            t1(std::max(c0e.x_min(),children[1]->bbox.x_min()),
               std::max(c0e.y_min(),children[1]->bbox.y_min()),
               std::max(c0e.z_min(),children[1]->bbox.z_min())),
            t2(std::min(c0e.x_max(),children[1]->bbox.x_max()),
               std::min(c0e.y_max(),children[1]->bbox.y_max()),
               std::min(c0e.z_max(),children[1]->bbox.z_max()));
        typename DISTBOXCLASS::Box3 ibox0(t1, t2);
        typename DISTBOXCLASS::Point3
            t3(std::max(c1e.x_min(),children[0]->bbox.x_min()),
               std::max(c1e.y_min(),children[0]->bbox.y_min()),
               std::max(c1e.z_min(),children[0]->bbox.z_min())),
            t4(std::min(c1e.x_max(),children[0]->bbox.x_max()),
               std::min(c1e.y_max(),children[0]->bbox.y_max()),
               std::min(c1e.z_max(),children[0]->bbox.z_max()));
        typename DISTBOXCLASS::Box3 ibox1(t3, t4);
        
				if (ibox0.x_length()*ibox0.y_length()*ibox0.z_length() < ibox1.x_length()*ibox1.y_length()*ibox1.z_length())
            addside = 0;
				else
            addside = 1;
        
			} else {
          // adding to neither would cause an intersection - add to the one that increases volume the least
				if (c0e.x_length()*c0e.y_length()*c0e.z_length() < c1e.x_length()*c1e.y_length()*c1e.z_length())
            addside = 0;
				else
            addside = 1;
			}
		} else if (children[0] && !children[1]) {
			addside = 0;
		} else if (!children[0] && children[1]) {
			addside = 1;
		}
    
      // expand our own bounding box
		bbox = (addside==-1 && objects.size()==0) ? obox : DISTBOXCLASS::Box3::make_union(bbox, obox);
    
		if (addside == -1) {
			objects.push_back(o);
			Split(boxclass, axis);
		} else {
			children[addside]->Insert(boxclass, o, (axis+1)%3);
		}
	}
  
	bool Remove(const DISTBOXCLASS &boxclass, const OBJECT &o) {
    
		if (bbox.classify_position(boxclass.bounding_box(o)) == DISTBOXCLASS::Box3::OUTSIDE)
        return false;
    
      // first check in the list of objects at this node
		for (unsigned i=0; i<objects.size(); i++) {
			if (o == objects[i]) {
        
          // remove the object from the list
				if (i != objects.size()-1) {
					objects[i] = objects.back();
				}
				objects.pop_back();
        
          // recompute the bounding box
				if (objects.size() > 0) {
          bbox = boxclass.bounding_box(objects[0]);
					for (unsigned i=1; i<objects.size(); i++) {
						bbox = DISTBOXCLASS::Box3::make_union(bbox, boxclass.bounding_box(objects[i]));
					}
				} else {
          typename DISTBOXCLASS::Point3
              u(-1,-1,-1),
              d(1,1,1);
          typename DISTBOXCLASS::Box3
              b(u, d);
          bbox = b;//DISTBOXCLASS::Box3(DISTBOXCLASS::Point3(-1,-1,-1), DISTBOXCLASS::Point3(1,1,1));
				}
        
				return true;
			}
		}
    
      // if we got here, we didn't find a match is the object list - check the children
		for (int c=0; c<2; c++) {
			if (children[c] && children[c]->Remove(boxclass, o)) {
				int dangle = children[c]->dangling();
				if (dangle != -1) {
          
            // the child we removed from now has no leaf objects and a single child - prune it from the tree
					BoxKDTree *child = children[c];
					children[c] = child->children[dangle];
					child->children[dangle] = NULL;
					delete child;
          
				} else if (children[c]->empty()) {
            // the child is now completely empty
					delete children[c];
					children[c] = NULL;
				}
				return true;
			}
		}
    
      // didn't find it anywhere!
		return false;
	}
  
	class OrderedTraverse {
    
public:
		OrderedTraverse(BoxKDTree &root, const typename DISTBOXCLASS::Point3 &p, const DISTBOXCLASS &_distclass) : distclass(_distclass) {
			point = p;
			pq.push(std::pair<double, std::pair<bool,void*> >(-root.bbox.distance(p), std::pair<bool,void*>(false,&root)));
		}
    
		typename DISTBOXCLASS::Box3::value_type next(OBJECT &o) {
      
			while (!pq.empty()) {
        
				std::pair<double, std::pair<bool,void*> > top = pq.top();
				pq.pop();
        
				if (top.second.first) {
            // it's a pointer to a triangle
					o = *(OBJECT*)top.second.second;
					return -top.first; // we negate the disance so the smallest is at the top of the heap
				} else {
            // otherwise it's a kdtree node index
					BoxKDTree *node = (BoxKDTree*)top.second.second;
          
            // add any objects that we might have in this node
					for (unsigned i=0; i<node->objects.size(); i++) {
						pq.push(std::pair<double, std::pair<bool,void*> >(-distclass.distance(node->objects[i], point), std::pair<bool,void*>(true,&node->objects[i])));
					}
          
            // add the children if they exist
					if (node->children[0]) {
						pq.push(std::pair<double, std::pair<bool,void*> >(-node->children[0]->bbox.distance(point), std::pair<bool,void*>(false,node->children[0])));
					}
          
					if (node->children[1]) {
						pq.push(std::pair<double, std::pair<bool,void*> >(-node->children[1]->bbox.distance(point), std::pair<bool,void*>(false,node->children[1])));
					}
				}
			}
			return -1;
		}
    
private:
		fast_pq< std::pair<double, std::pair<bool,void*> > >  pq;
		typename DISTBOXCLASS::Point3 point;
    
		const DISTBOXCLASS &distclass;
	};
  
private:
	void Split(const DISTBOXCLASS &boxclass, int axis=-1) {
    
      // check if we should stop splitting
		if (objects.size() >= BOXKDTREE_MAXLEAFSIZE) {
      
        // if we're not told what axis to use, use the biggest
			if (axis == -1) {
				typename DISTBOXCLASS::Box3::value_type xl, yl, zl;
				xl = bbox.x_length();
				yl = bbox.y_length();
				zl = bbox.z_length();
        
				if (xl>yl && xl>zl) axis = 0;
				else if (yl>xl && yl>zl) axis = 1;
				else axis = 2;
			}
      
        // split the list by the axis
			std::vector<OBJECT> cobjects[2];
      
			if (objects.size() < 500) {
				std::vector< std::pair<double,OBJECT> > sorter(objects.size());
        
				unsigned i;
				for (i=0; i<objects.size(); i++) {
					typename DISTBOXCLASS::Box3 obox = boxclass.bounding_box(objects[i]);
					sorter[i] = std::pair<double,OBJECT>((double)obox.centroid()[axis], objects[i]);
				}
        
				std::sort(sorter.begin(), sorter.end());
        
				for (i=0; i<sorter.size()/2; i++) {
					cobjects[0].push_back(sorter[i].second);
				}
				for ( ; i<sorter.size(); i++) {
					cobjects[1].push_back(sorter[i].second);
				}
        
			} else {
				for (unsigned i=0; i<objects.size(); i++) {
					typename DISTBOXCLASS::Box3 obox = boxclass.bounding_box(objects[i]);
          
					if (obox.centroid()[axis] < bbox.centroid()[axis]) {
						cobjects[0].push_back(objects[i]);
					} else {
						cobjects[1].push_back(objects[i]);
					}
				}
			}
      
			if ((cobjects[0].size() != 0) && (cobjects[1].size() != 0)) {
        
          // actually have to split
				objects.clear();
				ASSERT(!children[0] && !children[1]);
        
				children[0] = new BoxKDTree(cobjects[0], boxclass, (axis+1)%3);
				children[1] = new BoxKDTree(cobjects[1], boxclass, (axis+1)%3);
			}
		}
	}
  
	bool empty() {
		return (!children[0] && !children[1] && !objects.size());
	}
  
	int dangling() {
		if (!objects.size()) {
			if (children[0] && !children[1])	return 0;
			if (!children[0] && children[1])	return 1;
		}
		return -1;
	}
  
    // leaf node
	std::vector<OBJECT> objects;
  
    // internal node
	BoxKDTree* children[2];
  
	typename DISTBOXCLASS::Box3 bbox;
};

GTB_END_NAMESPACE

#endif // __KDTREE_H
