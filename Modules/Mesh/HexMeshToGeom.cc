
/*
 *  HexMeshToGeom.cc:  Convert a HexMesh into geoemtry
 *
 *  Written by:
 *   Peter Jensen
 *   Sourced from MeshToGeom.cc by David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   April 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Geom/Color.h>
#include <Geom/Geom.h>
#include <Geom/Group.h>
#include <Geom/Material.h>
#include <Geom/Line.h>
#include <Datatypes/HexMeshPort.h>
#include <Datatypes/ScalarFieldHUG.h>
#include <Datatypes/HexMesh.h>
#include <Malloc/Allocator.h>

class HexMeshToGeom : public Module {
    HexMeshIPort* imesh;
    GeometryOPort* ogeom;

    void mesh_to_geom(const HexMeshHandle&, GeomGroup*);
public:
    HexMeshToGeom(const clString& id);
    HexMeshToGeom(const HexMeshToGeom&, int deep);
    virtual ~HexMeshToGeom();
    virtual Module* clone(int deep);
    virtual void execute();
};

extern "C" {
Module* make_HexMeshToGeom(const clString& id)
{
    return scinew HexMeshToGeom(id);
}
}

HexMeshToGeom::HexMeshToGeom(const clString& id)
: Module("HexMeshToGeom", id, Filter)
{
    // Create the input port
    imesh=scinew HexMeshIPort(this, "HexMesh", HexMeshIPort::Atomic);
    add_iport(imesh);
    ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom);
}

HexMeshToGeom::HexMeshToGeom(const HexMeshToGeom&copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("HexMeshToGeom::HexMeshToGeom");
}

HexMeshToGeom::~HexMeshToGeom()
{
}

Module* HexMeshToGeom::clone(int deep)
{
    return scinew HexMeshToGeom(*this, deep);
}

void HexMeshToGeom::execute()
{
    HexMeshHandle mesh;    
    if (!imesh->get(mesh))
	return;

    int index, t, p1, p2;
    unsigned long long l;    
    HexFace * f;
    HashTable<unsigned long long, int> line_set;
    FourHexNodes c;
    GeomLines* lines = scinew GeomLines;
    
    for (index = mesh->high_face (); index >= 0; index--)
    {
      // Get a face.
    
      if ((f = mesh->find_face (index)) == NULL)
        continue;
    
      // Get the corners.
    
      c = f->corner_set ();
      
      // Loop through the pairs of corners.
      
      for (p1 = 0; p1 <= 3; p1++)
      {
        p2 = (p1 + 1) % 4 ;
        
        if (c.index[p1] == c.index[p2])
          continue;
        
        // Make a key for this pair of points.
          
        l = (c.index[p1] < c.index[p2]) ? (((unsigned long long) c.index[p1]) << 32) + c.index[p2]
                                        : (((unsigned long long) c.index[p2]) << 32) + c.index[p1];
        
        // If it's in the hash table, it's been added.
        
        if (line_set.lookup (l, t) != 0)
          continue;
          
        // Add the line segment.  
          
        line_set.insert (l, 1);
        
        lines->add (*(c.node[p1]), *(c.node[p2]));
      }
    }    
    
    GeomMaterial* matl=scinew GeomMaterial(lines,
					scinew Material(Color(0,0,0),
						     Color(0,.6,0), 
						     Color(.5,.5,.5), 20));
    ogeom->delAll();
    ogeom->addObj(matl, "Mesh1");
}
