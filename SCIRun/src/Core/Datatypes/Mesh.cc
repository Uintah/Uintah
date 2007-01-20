/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



#include <Core/Datatypes/Mesh.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun{

// initialize the static member type_id
PersistentTypeID Mesh::type_id("Mesh", "PropertyManager", NULL);


// A list to keep a record of all the different Field types that
// are supported through a virtual interface
Mutex MeshTypeIDMutex("Mesh Type ID Table Lock");
static std::map<string,MeshTypeID*>* MeshTypeIDTable = 0;

MeshTypeID::MeshTypeID(const string&type, MeshHandle (*mesh_maker)()) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(0),
    image_maker(0),
    scanline_maker(0),
    structhexvol_maker(0),
    structquadsurf_maker(0),
    structcurve_maker(0)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}

MeshTypeID::MeshTypeID(const string&type,
                         MeshHandle (*mesh_maker)(),
                         MeshHandle (*latvol_maker)(unsigned int x, unsigned int y, unsigned int z, const Point& min, const Point& max)
                         ) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(latvol_maker),
    image_maker(0),
    scanline_maker(0),
    structhexvol_maker(0),
    structquadsurf_maker(0),
    structcurve_maker(0)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}

MeshTypeID::MeshTypeID(const string&type,
                         MeshHandle (*mesh_maker)(),
                         MeshHandle (*image_maker)(unsigned int x, unsigned int y, const Point& min, const Point& max)
                         ) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(0),
    image_maker(image_maker),
    scanline_maker(0),
    structhexvol_maker(0),
    structquadsurf_maker(0),
    structcurve_maker(0)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}

MeshTypeID::MeshTypeID(const string&type,
                         MeshHandle (*mesh_maker)(),
                         MeshHandle (*scanline_maker)(unsigned int x, const Point& min, const Point& max)
                         ) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(0),
    image_maker(0),
    scanline_maker(scanline_maker),
    structhexvol_maker(0),
    structquadsurf_maker(0),
    structcurve_maker(0)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}


MeshTypeID::MeshTypeID(const string&type,
                         MeshHandle (*mesh_maker)(),
                         MeshHandle (*structhexvol_maker)(unsigned int x, unsigned int y, unsigned int z)
                         ) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(0),
    image_maker(0),
    scanline_maker(0),
    structhexvol_maker(structhexvol_maker),
    structquadsurf_maker(0),
    structcurve_maker(0)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}

MeshTypeID::MeshTypeID(const string&type,
                         MeshHandle (*mesh_maker)(),
                         MeshHandle (*structquadsurf_maker)(unsigned int x, unsigned int y)
                         ) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(0),
    image_maker(0),
    scanline_maker(0),
    structhexvol_maker(0),
    structquadsurf_maker(structquadsurf_maker),
    structcurve_maker(0)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}


MeshTypeID::MeshTypeID(const string&type,
                         MeshHandle (*mesh_maker)(),
                         MeshHandle (*structcurve_maker)(unsigned int x)
                         ) :
    type(type),
    mesh_maker(mesh_maker),
    latvol_maker(0),
    image_maker(0),
    scanline_maker(0),
    structhexvol_maker(0),
    structquadsurf_maker(0),
    structcurve_maker(structcurve_maker)
{
  MeshTypeIDMutex.lock();
  if (MeshTypeIDTable == 0)
  {
    MeshTypeIDTable = scinew std::map<string,MeshTypeID*>;
  }
  else
  {
    map<string,MeshTypeID*>::iterator dummy;
    
    dummy = MeshTypeIDTable->find(type);
    
    if (dummy != MeshTypeIDTable->end())
    {
      if ((*dummy).second->mesh_maker != mesh_maker)
      {
        std::cerr << "WARNING: duplicate mesh type exists: " << type << "\n";
        MeshTypeIDMutex.unlock();
        return;
      }
    }
  }
  std::cout << "Adding MeshTypeId :"<<type<<"\n";
  
  (*MeshTypeIDTable)[type] = this;
  MeshTypeIDMutex.unlock();
}

bool
Mesh::has_virtual_interface() const
{
  return (false);
}

Mesh::Mesh() :
  MIN_ELEMENT_VAL(1.0e-12)
{
}

Mesh::~Mesh() 
{
}

void 
Mesh::size(VNode::size_type& size) const
{
  size = 0;
}

void 
Mesh::size(VEdge::size_type& size) const
{
  size = 0;
}

void 
Mesh::size(VFace::size_type& size) const
{
  size = 0;
}

void 
Mesh::size(VCell::size_type& size) const
{
  size = 0;
}

void 
Mesh::size(VElem::size_type& size) const
{
  size = 0;
}

void 
Mesh::size(VDElem::size_type& size) const
{
  size = 0;
}
  
void 
Mesh::get_nodes(VNode::array_type& nodes, VNode::index_type i) const
{
  nodes.resize(1);
  nodes[0] = i;
}
  
void 
Mesh::get_nodes(VNode::array_type& nodes, VEdge::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_nodes(VNode::array_type,VEdge::index_type) has not been implemented");
}

void 
Mesh::get_nodes(VNode::array_type& nodes, VFace::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_nodes(VNode::array_type,VFace::index_type) has not been implemented");
}

void 
Mesh::get_nodes(VNode::array_type& nodes, VCell::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_nodes(VNode::array_type,VCell::index_type) has not been implemented");
}

void 
Mesh::get_nodes(VNode::array_type& nodes, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_nodes(VNode::array_type,VElem::index_type) has not been implemented");
}  

void 
Mesh::get_nodes(VNode::array_type& nodes, VDElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_nodes(VNode::array_type,VDElem::index_type) has not been implemented");
}



void 
Mesh::get_edges(VEdge::array_type& edges, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_edges(VEdge::array_type,VNode::index_type) has not been implemented");
}

void 
Mesh::get_edges(VEdge::array_type& edges, VEdge::index_type i) const
{
  edges.resize(1);
  edges[0] = i;
}

void 
Mesh::get_edges(VEdge::array_type& edges, VFace::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_edges(VEdge::array_type,VFace:index_type) has not been implemented");
}

void 
Mesh::get_edges(VEdge::array_type& edges, VCell::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_edges(VEdge::array_type,VCell::index_type) has not been implemented");
}

void 
Mesh::get_edges(VEdge::array_type& edges, VElem::index_type i) const
{  
  ASSERTFAIL("Mesh interface: get_edges(VEdge::array_type,VElem::index_type) has not been implemented");
}

void 
Mesh::get_edges(VEdge::array_type& edges, VDElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_edges(VEdge::array_type,VDElem::index_type) has not been implemented");
}


void 
Mesh::get_faces(VFace::array_type& faces, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_faces(VFace::array_type,VNode::index_type) has not been implemented");
}

void 
Mesh::get_faces(VFace::array_type& faces, VEdge::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_faces(VFace::array_type,VEdge::index_type) has not been implemented");
}

void 
Mesh::get_faces(VFace::array_type& faces, VFace::index_type i) const
{
  faces.resize(1);
  faces[0] = i;
}

void 
Mesh::get_faces(VFace::array_type& faces, VCell::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_faces(VFace::array_type,VCell::index_type) has not been implemented");
}

void 
Mesh::get_faces(VFace::array_type& faces, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_faces(VFace::array_type,VElem::index_type) has not been implemented");
}

void 
Mesh::get_faces(VFace::array_type& faces, VDElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_faces(VFace::array_type,VDElem::index_type) has not been implemented");
}



void 
Mesh::get_cells(VCell::array_type& cells, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_cells(VCell::array_type,VNode::index_type) has not been implemented");
}

void 
Mesh::get_cells(VCell::array_type& cells, VEdge::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_cells(VCell::array_type,VEdge::index_type) has not been implemented");
}

void 
Mesh::get_cells(VCell::array_type& cells, VFace::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_cells(VCell::array_type,VFace::index_type) has not been implemented");
}

void 
Mesh::get_cells(VCell::array_type& cells, VCell::index_type i) const
{
  cells.resize(1);
  cells[0] = i;
}

void 
Mesh::get_cells(VCell::array_type& cells, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_cells(VCell::array_type,VElem::index_type) has not been implemented");
}

void 
Mesh::get_cells(VCell::array_type& cells, VDElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_cells(VCell::array_type,VDElem::index_type) has not been implemented");
}

  
  
void 
Mesh::get_elems(VElem::array_type& elems, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_elems(VElem::array_type,VNode::index_type) has not been implemented");
}

void 
Mesh::get_elems(VElem::array_type& elems, VEdge::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_elems(VElem::array_type,VEdge::index_type) has not been implemented");
}

void 
Mesh::get_elems(VElem::array_type& elems, VFace::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_elems(VElem::array_type,VFace::index_type) has not been implemented");
}

void 
Mesh::get_elems(VElem::array_type& elems, VCell::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_elems(VElem::array_type,VCell::index_type) has not been implemented");
}

void 
Mesh::get_elems(VElem::array_type& elems, VElem::index_type i) const
{
  elems.resize(1);
  elems[0] = i;
}

void 
Mesh::get_elems(VElem::array_type& elems, VDElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_elems(VElem::array_type,VDElem::index_type) has not been implemented");
}



void 
Mesh::get_delems(VDElem::array_type& delems, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_delems(VDElem::array_type,VNode::index_type) has not been implemented");
}

void 
Mesh::get_delems(VDElem::array_type& delems, VEdge::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_delems(VDElem::array_type,VEdge::index_type) has not been implemented");
}

void 
Mesh::get_delems(VDElem::array_type& delems, VFace::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_delems(VDElem::array_type,VFace::index_type) has not been implemented");
}

void 
Mesh::get_delems(VDElem::array_type& delems, VCell::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_delems(VDElem::array_type,VCell::index_type) has not been implemented");
}

void 
Mesh::get_delems(VDElem::array_type& delems, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_delems(VDElem::array_type,VElem::index_type) has not been implemented");
}

void 
Mesh::get_delems(VDElem::array_type& delems, VDElem::index_type i) const
{
  delems.resize(1);
  delems[0] = i;
}

void 
Mesh::get_center(Point &point, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_center(Point,VNode::index_type) has not been implemented");
}

void 
Mesh::get_center(Point &point, VEdge::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_center(Point,VEdge::index_type) has not been implemented");
}

void 
Mesh::get_center(Point &point, VFace::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_center(Point,VFace::index_type) has not been implemented");
}

void 
Mesh::get_center(Point &point, VCell::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_center(Point,VCell::index_type) has not been implemented");
}

void 
Mesh::get_center(Point &point, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_center(Point,VElem::index_type) has not been implemented");
}

void 
Mesh::get_center(Point &point, VDElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_center(Point,VDElem::index_type) has not been implemented");
}


double 
Mesh::get_size(Mesh::VNode::index_type i) const
{
  return (0.0);
}

double 
Mesh::get_size(Mesh::VEdge::index_type i) const
{
  return (0.0);
}

double 
Mesh::get_size(Mesh::VFace::index_type i) const
{
  return (0.0);
} 

double 
Mesh::get_size(Mesh::VCell::index_type i) const
{
  return (0.0);
}

double 
Mesh::get_size(Mesh::VElem::index_type i) const
{
  return (0.0);
}

double 
Mesh::get_size(Mesh::VDElem::index_type i) const
{
  return (0.0);
}
  
  
void 
Mesh::get_weights(const Point& p,VNode::array_type& nodes,
                                                vector<double>& weights) const
{
  ASSERTFAIL("Mesh interface: get_weights has not been implemented for nodes");
}

void 
Mesh::get_weights(const Point& p,VElem::array_type& elems,
                                                vector<double>& weights) const
{
  ASSERTFAIL("Mesh interface: get_weights has not been implemented for nodes");
}



bool 
Mesh::locate(VNode::index_type &i, const Point &point) const
{
  ASSERTFAIL("Mesh interface: locate(VNode::index_type,Point) has not been implemented");
}

bool 
Mesh::locate(VElem::index_type &i, const Point &point) const
{
  ASSERTFAIL("Mesh interface: locate(VElem::index_type,Point) has not been implemented");
}

  
bool 
Mesh::get_coords(vector<double> &coords, const Point &point, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_coords(vector<double>,Point,VElem::index_type) has not been implemented");
}


void 
Mesh::interpolate(Point &p, const vector<double> &coords, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: interpolate(Point,vector<double>,VElem::index_type) has not been implemented");
}


void 
Mesh::derivate(vector<Point> &p, const vector<double> &coords, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: derivate(vector<Point>,vector<double>,VElem::index_type) has not been implemented");
}


void 
Mesh::get_normal(Vector &result, vector<double> &coords, VElem::index_type eidx, unsigned int f) const
{
  ASSERTFAIL("Mesh interface: get_normal() has not been implemented");
}  


void 
Mesh::get_points(vector<Point> &points) const
{
  ASSERTFAIL("Mesh interface: get_points(vector<Point>&) has not been implemented");
}


void 
Mesh::get_random_point(Point &point, VElem::index_type i,MusilRNG &rng) const
{
  ASSERTFAIL("Mesh interface: get_random_point(Point,VElem::index_type) has not been implemented");
}

void 
Mesh::set_point(const Point &point, VNode::index_type i)
{
  ASSERTFAIL("Mesh interface: set_point(Point,VNode::index_type) has not been implemented");
}
  
  

void 
Mesh::node_reserve(size_t size)
{
  ASSERTFAIL("Mesh interface: node_reserve(size_t size) has not been implemented");
}

void 
Mesh::elem_reserve(size_t size)
{
  ASSERTFAIL("Mesh interface: elem_reserve(size_t size) has not been implemented");
}


void 
Mesh::add_node(const Point &point, VNode::index_type &i)
{
  ASSERTFAIL("Mesh interface: this mesh cannot be edited (add_node)");  
}

void  
Mesh::add_elem(const VNode::array_type &nodes, VElem::index_type &i)
{
  ASSERTFAIL("Mesh interface: this mesh cannot be edited (add_elem)");  
}

bool
Mesh::get_neighbor(VElem::index_type &neighbor, VElem::index_type from, VDElem::index_type delem) const
{
  ASSERTFAIL("Mesh interface: get_neighbor(VElem::index_type,VElem::index_type,VDElem::index_type) has not been implemented");  
}

void
Mesh::get_neighbors(VElem::array_type &elems, VElem::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_neighbors(VElem::index_type,VElem::index_type) has not been implemented");  
}

void
Mesh::get_neighbors(VNode::array_type &nodes, VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface: get_neighbors(VNode::index_type,VNode::index_type) has not been implemented");  
}

void 
Mesh::pwl_approx_edge(vector<vector<double> > &coords, VElem::index_type ci, unsigned int which_edge, unsigned int div_per_unit) const
{
  ASSERTFAIL("Mesh interface: pwl_appprox_edge has not been implemented");  
}

void 
Mesh::pwl_approx_face(vector<vector<vector<double> > > &coords, VElem::index_type ci, unsigned int which_face, unsigned int div_per_unit) const
{
  ASSERTFAIL("Mesh interface: pwl_appprox_face has not been implemented");  
}

void 
Mesh::get_normal(Vector& norm,VNode::index_type i) const
{
  ASSERTFAIL("Mesh interface:get_normal has not been implemented");  
}

int
Mesh::basis_order()
{
  return (-1);
}

void
Mesh::get_dimensions(dimension_type& dim)
{
  dim.resize(1);
  VNode::size_type sz;
  size(sz);
  dim[0] = sz;
}


const int MESHBASE_VERSION = 2;

void 
Mesh::io(Piostream& stream)
{
  if (stream.reading() && stream.peek_class() == "MeshBase")
  {
    stream.begin_class("MeshBase", 1);
  }
  else
  {
    stream.begin_class("Mesh", MESHBASE_VERSION);
  }
  PropertyManager::io(stream);
  stream.end_class();
}

const string 
Mesh::type_name(int n)
{
  ASSERT(n >= -1 && n <= 0);
  static const string name = "Mesh";
  return name;
}

//! Return the transformation that takes a 0-1 space bounding box 
//! to the current bounding box of this mesh.
void Mesh::get_canonical_transform(Transform &t) 
{
  t.load_identity();
  BBox bbox = get_bounding_box();
  t.pre_scale(bbox.diagonal());
  t.pre_translate(Vector(bbox.min()));
}



MeshHandle
Create_Mesh(string type)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->mesh_maker != 0)
    {
      handle = (*it).second->mesh_maker();
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}

MeshHandle
Create_Mesh(string type,unsigned int x, unsigned int y, unsigned int z, const Point& min, const Point& max)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->latvol_maker != 0)
    {
      handle = (*it).second->latvol_maker(x,y,z,min,max);
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}

MeshHandle
Create_Mesh(string type,unsigned int x, unsigned int y,const Point& min, const Point& max)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->image_maker != 0)
    {
      handle = (*it).second->image_maker(x,y,min,max);
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}

MeshHandle
Create_Mesh(string type,unsigned int x, const Point& min, const Point& max)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->scanline_maker != 0)
    {
      handle = (*it).second->scanline_maker(x,min,max);
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}

MeshHandle
Create_Mesh(string type,unsigned int x, unsigned int y, unsigned int z)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->structhexvol_maker != 0)
    {
      handle = (*it).second->structhexvol_maker(x,y,z);
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}

MeshHandle
Create_Mesh(string type,unsigned int x, unsigned int y)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->structquadsurf_maker != 0)
    {
      handle = (*it).second->structquadsurf_maker(x,y);
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}

MeshHandle
Create_Mesh(string type,unsigned int x)
{
  MeshHandle handle(0);
  MeshTypeIDMutex.lock();
  std::map<string,MeshTypeID*>::iterator it;
  it = MeshTypeIDTable->find(type);
  if (it != MeshTypeIDTable->end()) 
  {
    if ((*it).second->structcurve_maker != 0)
    {
      handle = (*it).second->structcurve_maker(x);  
    }
  }
  MeshTypeIDMutex.unlock();
  return (handle);
}




} // end namespace
