//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : InterfaceWithTetGen.h
//    Author : Martin Cole
//    Date   : Thu Mar 23 10:17:04 2006

#if !defined(InterfaceWithTetGen_h)
#define InterfaceWithTetGen_h


#include <Dataflow/Network/Module.h>
#include <Dataflow/Modules/Fields/InterfaceWithTetGen.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Basis/Constant.h>

#include <tetgen.h>

namespace SCIRun {



class InterfaceWithTetGenInterface : public DynamicAlgoBase
{
public:
  virtual ~InterfaceWithTetGenInterface() {}

  static 
  CompileInfoHandle get_compile_info(const TypeDescription *td, 
				     const string template_class_name);
  
};


class TGRegionAttribAlgo : public InterfaceWithTetGenInterface
{
public:
  virtual ~TGRegionAttribAlgo() {}

  //! given the input field generate the appropriate data in the tetgenio.
  virtual void set_region_attribs(ProgressReporter *pr, FieldHandle, 
				  tetgenio &) = 0;
};

template <class Fld>
class TGRegionAttrib : public TGRegionAttribAlgo
{
public:
  //! given the input field generate the appropriate data in the tetgenio.
  virtual void set_region_attribs(ProgressReporter *pr, FieldHandle, 
				  tetgenio &);
};


class TGAdditionalPointsAlgo : public InterfaceWithTetGenInterface
{
public:
  virtual ~TGAdditionalPointsAlgo() {}
  
  //! given the input fields generate the appropriate data in the tetgenio.
  virtual void add_points(ProgressReporter        *pr, 
			  FieldHandle              o, 
			  tetgenio                &in) = 0;
};

template <class Fld>
class TGAdditionalPoints : public TGAdditionalPointsAlgo
{
public:
  virtual ~TGAdditionalPoints() {}
  
  //! given the input fields generate the appropriate data in the tetgenio.
  virtual void add_points(ProgressReporter        *pr, 
			  FieldHandle              o, 
			  tetgenio                &in);
};

class TGSurfaceTGIOAlgo : public InterfaceWithTetGenInterface
{
public:
  virtual ~TGSurfaceTGIOAlgo() {}
  
  //! given the input fields generate the appropriate data in the tetgenio.
  virtual void to_tetgenio(ProgressReporter        *pr, 
			   FieldHandle              o, 
			   unsigned                &idx,
			   unsigned                &fidx,
			   int                      region_id,
			   tetgenio                &in) = 0;

  //! given the specified tetrahedral volume in the input dat, return 
  //! a TetVol Field.
  virtual FieldHandle to_tetvol(const tetgenio &dat) = 0;

};

template <class Fld>
class TGSurfaceTGIO : public TGSurfaceTGIOAlgo
{
public:
  //! given the input fields generate the appropriate data in the tetgenio.
  virtual void to_tetgenio(ProgressReporter        *pr, 
			   FieldHandle              o, 
			   unsigned                &idx,
			   unsigned                &fidx,
			   int                      region_id,
			   tetgenio                &in);

  //! given the specified tetrahedral volume in the input dat, return 
  //! a TetVol Field.
  virtual FieldHandle to_tetvol(const tetgenio &dat);
};

template <class Msh>
void
add_surface_info(Msh *mesh, tetgenio &in, unsigned &idx, unsigned &fidx, 
		 const int marker)
{
  typedef typename Msh::basis_type Bas;

  unsigned off = idx;
  //iterate over nodes and add the points.
  typename Msh::Node::iterator ni, end;
  mesh->begin(ni);
  mesh->end(end);
  while (ni != end) {
    Point p;
    mesh->get_center(p, *ni);
    in.pointlist[idx * 3] = p.x();
    in.pointlist[idx * 3 + 1] = p.y();
    in.pointlist[idx * 3 + 2] = p.z();
    ++ni; ++idx;
  }

  typename Msh::Face::size_type fsz;
  const unsigned vert_per_face = Bas::vertices_of_face();

  // iterate over faces.
  typename Msh::Face::iterator fi, fend;
  mesh->begin(fi);
  mesh->end(fend);
  while (fi != fend) {
    tetgenio::facet *f = &in.facetlist[fidx];
    f->numberofpolygons = 1;
    f->polygonlist = new tetgenio::polygon[f->numberofpolygons];
    f->numberofholes = 0;
    f->holelist = 0;
    tetgenio::polygon *p = &f->polygonlist[0];
    p->numberofvertices = vert_per_face;
    p->vertexlist = new int[p->numberofvertices];
    
    typename Msh::Node::array_type nodes;
    mesh->get_nodes(nodes, *fi);
    typename Msh::Node::array_type::iterator niter = nodes.begin();
    unsigned nidx = 0;
    while (niter != nodes.end()) {
      p->vertexlist[nidx++] = *niter++ + off;
    }

    in.facetmarkerlist[fidx] = marker;
    ++fi; ++fidx;
  }
}

template <class Fld>
void
TGSurfaceTGIO<Fld>::to_tetgenio(ProgressReporter        *pr, 
				FieldHandle              o, 
				unsigned                &idx,
				unsigned                &fidx,
				int                      region_id,
				tetgenio                &in)
{
  Fld* outer = dynamic_cast<Fld*>(o.get_rep());
  typedef typename Fld::mesh_type Msh;
  Msh *mesh = outer->get_typed_mesh().get_rep();

  typename Msh::Node::size_type nsz;
  typename Msh::Face::size_type fsz;

  mesh->size(nsz);
  mesh->size(fsz);

  REAL *tmppl = 0;
  if (in.pointlist) {
    tmppl = in.pointlist;
  }
  in.pointlist = new REAL[(nsz + in.numberofpoints) * 3];
  if (tmppl) {
    memcpy(in.pointlist, tmppl, sizeof(REAL) * in.numberofpoints * 3);
    delete[] tmppl;
  }
  in.numberofpoints += nsz;

  tetgenio::facet *tmpfl = 0;
  int *tmpfml = 0;
  if (in.facetlist) {
    tmpfl = in.facetlist;
    tmpfml = in.facetmarkerlist;
  }
  
  in.facetlist = new tetgenio::facet[in.numberoffacets + fsz];
  in.facetmarkerlist = new int[in.numberoffacets + fsz];
  if (tmpfl) {
    memcpy(in.facetlist, tmpfl, sizeof(tetgenio::facet) * in.numberoffacets);
    memcpy(in.facetmarkerlist, tmpfml, sizeof(int) * in.numberoffacets);
    delete[] tmpfl;
    delete[] tmpfml;
  }
  in.numberoffacets += fsz;

  add_surface_info(mesh, in, idx, fidx, region_id);
}

template <class Fld>
FieldHandle 
TGSurfaceTGIO<Fld>::to_tetvol(const tetgenio &dat)
{
  typedef TetVolMesh<TetLinearLgn<Point> > TVMesh;
  typedef GenericField<TVMesh, ConstantBasis<double>, vector<double> > TVField;
  TVMesh *mesh = new TVMesh();


  for (int i = 0; i < dat.numberofpoints; i++) {
    Point p(dat.pointlist[i*3], dat.pointlist[i*3+1], dat.pointlist[i*3+2]);
    mesh->add_point(p);
  }  

  for (int i = 0; i < dat.numberoftetrahedra; i++) {
    mesh->add_tet(dat.tetrahedronlist[i*4], dat.tetrahedronlist[i*4+1], 
		  dat.tetrahedronlist[i*4+2], dat.tetrahedronlist[i*4+3]);
  }
  
  TVField *tvf = new TVField(LockingHandle<TVMesh>(mesh));
  tvf->resize_fdata();

  int atts =  dat.numberoftetrahedronattributes;
  for (int i = 0; i < dat.numberoftetrahedra; i++) {
     for (int j = 0; j < atts; j++) {
       double val = dat.tetrahedronattributelist[i * atts + j];
       typename TVMesh::Elem::index_type idx = i;
       tvf->set_value(val, idx);
    }
  }   
  FieldHandle fh(tvf);
  return fh;
}


template <class Fld>
void
TGRegionAttrib<Fld>::set_region_attribs(ProgressReporter *pr, FieldHandle ra, 
					tetgenio &in)
{
  Fld* rattribs = dynamic_cast<Fld*>(ra.get_rep());
  if (! rattribs) {
    pr->error("Passed in Field type does not match compiled type.");
    return;
  }
  typedef typename Fld::mesh_type Msh;
  typedef typename Msh::basis_type Bas;
  Msh *mesh = rattribs->get_typed_mesh().get_rep();

  //iterate over nodes and add the points.
  typename Msh::Node::iterator ni, end;
  mesh->begin(ni);
  mesh->end(end);
  unsigned idx = 0;
  typename Msh::Node::size_type sz;
  mesh->size(sz);
  // Allocate the list.
  in.regionlist = new REAL[sz*5];
  in.numberofregions = sz;
  while (ni != end) {
    Point p;
    mesh->get_center(p, *ni);
    in.regionlist[idx * 5] = p.x();
    in.regionlist[idx * 5 + 1] = p.y();
    in.regionlist[idx * 5 + 2] = p.z();
    in.regionlist[idx * 5 + 3] = idx;
    typename Fld::value_type val;
    rattribs->value(val, *ni);
    in.regionlist[idx * 5 + 4] = val;
    ++ni; ++idx;
  }

}


template <class Fld>
void 
TGAdditionalPoints<Fld>::add_points(ProgressReporter        *pr, 
				    FieldHandle              ap, 
				    tetgenio                &in)
{
  Fld* addp = dynamic_cast<Fld*>(ap.get_rep());
  if (! addp) {
    pr->error("Passed in Field type does not match compiled type.");
    return;
  }
  typedef typename Fld::mesh_type Msh;
  typedef typename Msh::basis_type Bas;
  Msh *mesh = addp->get_typed_mesh().get_rep();

  //NOTE: in 1.4.2 tetgen the addpointlist goes away. We will be adding an
  //      additional tetgenio structure somehow.

  //iterate over nodes and add the points.
  typename Msh::Node::iterator ni, end;
  mesh->begin(ni);
  mesh->end(end);
  unsigned idx = 0;
  typename Msh::Node::size_type sz;
  mesh->size(sz);
  in.numberofaddpoints = sz;
  // Allocate the list.
  in.addpointlist = new REAL[sz*3];
  while (ni != end) {
    Point p;
    mesh->get_center(p, *ni);
    in.addpointlist[idx * 3] = p.x();
    in.addpointlist[idx * 3 + 1] = p.y();
    in.addpointlist[idx * 3 + 2] = p.z();

    ++ni; ++idx;
  }
}

} //namespace SCIRun

#endif //InterfaceWithTetGen_h
