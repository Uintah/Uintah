
/*
 *  Group.cc:  Groups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Geom/Group.h>
#include <Classlib/Array2.h>
#include <Malloc/New.h>
#include <values.h>

GeomGroup::GeomGroup(int del_children)
: GeomObj(), objs(0, 100), del_children(del_children), treetop(0)
{
}

GeomGroup::GeomGroup(const GeomGroup& copy)
: GeomObj(copy), bb(copy.bb), del_children(copy.del_children), treetop(0)
{
    objs.grow(copy.objs.size());
    for(int i=0;i<objs.size();i++){
	GeomObj* cobj=copy.objs[i];
	objs[i]=cobj->clone();
	objs[i]->set_parent(this);
    }
}

GeomGroup::~GeomGroup()
{
    if(treetop)
	delete treetop;
    if(del_children){
	for(int i=0;i<objs.size();i++)
	    delete objs[i];
    }
}

void GeomGroup::add(GeomObj* obj)
{
    obj->set_parent(this);
    objs.add(obj);
}

void GeomGroup::remove(GeomObj* obj)
{
   for(int i=0;i<objs.size();i++)
      if (objs[i] == obj) {
	 objs.remove(i);
	 if(del_children)delete obj;
	 break;
      }
}

void GeomGroup::remove_all()
{
   if(del_children)
      for(int i=0;i<objs.size();i++)
	 delete objs[i];
   objs.remove_all();
}

int GeomGroup::size()
{
    return objs.size();
}

GeomObj* GeomGroup::clone()
{
    return new GeomGroup(*this);
}

void GeomGroup::get_bounds(BBox& in_bb)
{
    if(1 || !bb.valid()){
	for(int i=0;i<objs.size();i++)
	    objs[i]->get_bounds(bb);
    }
    if(bb.valid())
	in_bb.extend(bb);
}

void GeomGroup::get_bounds(BSphere& in_sphere)
{
    if(!bsphere.valid()){
	for(int i=0;i<objs.size();i++)
	    objs[i]->get_bounds(bsphere);
    }
    if(bsphere.valid())
	in_sphere.extend(bsphere);
}


void GeomGroup::make_prims(Array1<GeomObj*>& free,
			 Array1<GeomObj*>& dontfree)
{
    for(int i=0;i<objs.size();i++){
	MemoryManager::audit(objs[i]);
	objs[i]->make_prims(free, dontfree);
    }
}

void GeomGroup::reset_bbox()
{
    if(treetop){
	delete treetop;
	treetop=0;
    }
    for(int i=0;i<objs.size();i++)
	objs[i]->reset_bbox();
    bb.reset();
}

void GeomGroup::preprocess()
{
    for(int i=0;i<objs.size();i++){
	objs[i]->preprocess();
    }

    // Build the tree...
    if(treetop)
	delete treetop;
    int s=Min(objs.size(), 100);
    Array2<double> volumes(s, s);
    Array1<ITree*> current(s);
    Array1<BSphere> bounds(s);
    for(i=0;i<s;i++){
	current[i]=new ITreeLeaf(objs[i]);
	volumes(i, i)=MAXDOUBLE;
	objs[i]->get_bounds(bounds[i]);
	for(int j=0;j<i;j++){
	    BSphere bs(bounds[i]);
	    bs.extend(bounds[j]);
	    volumes(i,j)=volumes(j,i)=bs.volume();
	}
    }
    int next=s;
    ITree* newnode=current[0];
    for(int count=1;count<objs.size();count++){
	// Fix this loop - cache minimums for each row.
	double min=MAXDOUBLE;
	int obj1=-1, obj2=-1;
	for(int i=0;i<s;i++){
	    for(int j=0;j<i;j++){
		if(volumes(i,j) < min){
		    min=volumes(i,j);
		    obj1=i;
		    obj2=j;
		}
	    }
	}
	// Found a candidate, join these two...

	// Decide if it should have the BV or not
	double v1=current[obj1]->volume;
	double v2=current[obj2]->volume;
	double vwhole=min;
	if(v1*1.1 < vwhole || v2*1.1 < vwhole){
	    // Volume changes a lot, make this one a BSphere node
	    newnode=new ITreeNodeBSphere(current[obj1], current[obj2],
					 bounds[obj1], bounds[obj2]);
	    newnode->volume=vwhole;
	} else {
	    newnode=new ITreeNode(current[obj1], current[obj2]);
	    newnode->volume=Min(v1, v2);
	}
	// Fix up the arrays...
	current[obj1]=newnode;
	bounds[obj1].extend(bounds[obj2]);
	if(next < objs.size()){
	    current[obj2]=new ITreeLeaf(objs[next]);
	    objs[next]->get_bounds(bounds[obj2]);
	    for(int i=0;i<s;i++){
		if(current[i] && i!=obj2){
		    BSphere s(bounds[obj2]);
		    s.extend(bounds[i]);
		    volumes(obj2,i)=volumes(i,obj2)=s.volume();
		} else {
		    volumes(obj2,i)=volumes(i,obj2)=MAXDOUBLE;
		}
	    }
	    next++;
	} else {
	    for(int i=0;i<s;i++){
		volumes(obj2, i)=volumes(i,obj2)=MAXDOUBLE;
	    }
	    current[obj2]=0;
	    bounds[obj2].reset();
	}
	for(i=0;i<s;i++){
	    if(current[i] && i!=obj1){
		BSphere s(bounds[obj1]);
		s.extend(bounds[i]);
		volumes(obj1,i)=volumes(i,obj1)=s.volume();
	    } else {
		volumes(obj1,i)=volumes(i,obj1)=MAXDOUBLE;
	    }
	}
    }
    // The top of the tree will be in newnode
    treetop=newnode;
}

void GeomGroup::intersect(const Ray& ray, Material* matl,
			  Hit& hit)
{
    treetop->intersect(ray, matl, hit);
#if 0
    for(int i=0;i<objs.size();i++){
	objs[i]->intersect(ray, matl, hit);
    }
#endif
}

GeomGroup::ITreeLeaf::ITreeLeaf(GeomObj* obj)
: obj(obj)
{
}

GeomGroup::ITreeLeaf::~ITreeLeaf()
{
}

void GeomGroup::ITreeLeaf::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    obj->intersect(ray, matl, hit);
}

GeomGroup::ITreeNode::ITreeNode(GeomGroup::ITree* left,
				GeomGroup::ITree* right)
: left(left), right(right)
{
}

GeomGroup::ITreeNode::~ITreeNode()
{
    delete left;
    delete right;
}

void GeomGroup::ITreeNode::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    left->intersect(ray, matl, hit);
    right->intersect(ray, matl, hit);
}

GeomGroup::ITreeNodeBSphere::ITreeNodeBSphere(GeomGroup::ITree* left,
					      GeomGroup::ITree* right,
					      const BSphere& lbound,
					      const BSphere& rbound)
: left(left), right(right), bsphere(lbound)
{
    bsphere.extend(rbound);
}

GeomGroup::ITreeNodeBSphere::~ITreeNodeBSphere()
{
    delete left;
    delete right;
}

void GeomGroup::ITreeNodeBSphere::intersect(const Ray& ray, Material* matl, Hit& hit)
{
    if(!bsphere.intersect(ray))
	return;
    left->intersect(ray, matl, hit);
    right->intersect(ray, matl, hit);
}

GeomGroup::ITree::~ITree()
{
}
