
/*
 *  Group.h:  Groups of GeomObj's
 *
 *  Written by:
 *   Steven G. Parker & David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_Group_h
#define SCI_Geom_Group_h 1

#include <Geom/Geom.h>
#include <Geometry/BBox.h>
#include <Geometry/BSphere.h>

class GeomGroup : public GeomObj {
    Array1<GeomObj*> objs;
    BBox bb;
    BSphere bsphere;
    int del_children;

public:
    struct ITree {
	double volume;
	virtual ~ITree();
	virtual void intersect(const Ray& ray, Material* matl, Hit& hit)=0;
    };
    struct ITreeLeaf : public ITree {
	GeomObj* obj;
	ITreeLeaf(GeomObj*);
	virtual ~ITreeLeaf();
	virtual void intersect(const Ray& ray, Material* matl, Hit& hit);
    };
    struct ITreeNode : public ITree {
	ITree* left;
	ITree* right;
	ITreeNode(ITree*, ITree*);
	virtual ~ITreeNode();
	virtual void intersect(const Ray& ray, Material* matl, Hit& hit);
    };
    struct ITreeNodeBSphere : public ITree {
	ITree* left;
	ITree* right;
	BSphere bsphere;
	ITreeNodeBSphere(ITree*, ITree*, const BSphere&, const BSphere&);
	virtual ~ITreeNodeBSphere();
	virtual void intersect(const Ray& ray, Material* matl, Hit& hit);
    };
private:
    ITree* treetop;
public:
    GeomGroup(int del_children=1);
    GeomGroup(const GeomGroup&);
    virtual ~GeomGroup();
    virtual GeomObj* clone();

    void add(GeomObj*);
    int size();

    virtual void reset_bbox();
    virtual void get_bounds(BBox&);
    virtual void get_bounds(BSphere&);

#ifdef SCI_OPENGL
    virtual void draw(DrawInfoOpenGL*, Material*);
#endif
    virtual void make_prims(Array1<GeomObj*>& free,
			    Array1<GeomObj*>& dontfree);
    virtual void preprocess();
    virtual void intersect(const Ray& ray, Material*,
			   Hit& hit);
};

#endif /* SCI_Geom_Group_h */
