#include <Geom/IndexedGroup.h>
#include <iostream.h>


GeomIndexedGroup::GeomIndexedGroup(const GeomIndexedGroup& /* g */)
{
    cerr << "GeomIndexedGroup::GeomIndexedGroup(const GeomIndexedGroup&) not implemented!\n";
}

GeomIndexedGroup::GeomIndexedGroup()
{
    // do nothing for now
}

GeomIndexedGroup::~GeomIndexedGroup()
{
    delAll();  // just nuke everything for now...
}

GeomObj* GeomIndexedGroup::clone()
{
    cerr << "GeomObj* GeomIndexedGroup::clone() not implemented!\n";
}

void GeomIndexedGroup::reset_bbox()
{
    cerr << "void GeomIndexedGroup::reset_bbox() not implemented!\n";
}


void GeomIndexedGroup::get_bounds(BBox& bbox)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->get_bounds(bbox);
    }
}

void GeomIndexedGroup::get_bounds(BSphere& bsphere)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->get_bounds(bsphere);
    }
}

void GeomIndexedGroup::make_prims(Array1<GeomObj*>& free,
				  Array1<GeomObj*>& dontfree)
{
    HashTableIter<int, GeomObj*> iter(&objs);   
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->make_prims(free,dontfree);
    }	
}	

void GeomIndexedGroup::preprocess()
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->preprocess();
    }
}

void GeomIndexedGroup::intersect(const Ray& ray, Material* m,
				 Hit& hit)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	obj->intersect(ray, m, hit);
    }
}

void GeomIndexedGroup::io(Piostream&)
{
    cerr << "GeomIndexedGroup::io not implemented!\n";
}	

void GeomIndexedGroup::addObj(GeomObj* obj, int id)
{
    objs.insert(id,obj);
}

GeomObj* GeomIndexedGroup::getObj(int id)
{
    GeomObj *obj;
    if (objs.lookup(id,obj)) {
	return obj;
    }
    else {
	cerr << "couldn't find object in GeomIndexedGroup::getObj!\n";
    }
    return 0;
}

void GeomIndexedGroup::delObj(int id)
{
    GeomObj* obj;

    if (objs.lookup(id,obj)) {
	objs.remove(id);
	delete obj;
    }
    else {
	cerr << "invalid id in GeomIndexedGroup::delObj()!\n";
    }
}

void GeomIndexedGroup::delAll(void)
{
    HashTableIter<int, GeomObj*> iter(&objs);
    for(iter.first();iter.ok();++iter) {
	GeomObj *obj = iter.get_data();
	delete obj;
    }

    objs.remove_all();
}

HashTableIter<int,GeomObj*> GeomIndexedGroup::getIter(void)
{
    HashTableIter<int,GeomObj*> iter(&objs);
    return iter;
}

HashTable<int,GeomObj*>* GeomIndexedGroup::getHash(void)
{
    return &objs;
}

#ifdef __GNUG__

#include <Classlib/HashTable.cc>
template class HashTable<int, GeomObj*>;
template class HashTableIter<int, GeomObj*>;
template class HashKey<int, GeomObj*>;

#endif
