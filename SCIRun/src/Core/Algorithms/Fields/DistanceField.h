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

#ifndef CORE_ALGORITHMS_FIELDS_FIELDDISTANCE_H
#define CORE_ALGORITHMS_FIELDS_FIELDDISTANCE_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <float.h>

namespace SCIRunAlgo {

using namespace SCIRun;

class DistanceFieldCellAlgo : public DynamicAlgoBase
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object, FieldHandle objectboundary);
};

template<class FSRC, class FDST, class FOBJ, class DFOBJ>
class DistanceFieldCellAlgoT : public DistanceFieldCellAlgo
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object, FieldHandle objectboundary);
};


class DistanceFieldFaceAlgo : public DynamicAlgoBase
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};


template<class FSRC, class FDST, class FOBJ>
class DistanceFieldFaceAlgoT : public DistanceFieldFaceAlgo
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};

class SignedDistanceFieldFaceAlgo : public DynamicAlgoBase
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};


template<class FSRC, class FDST, class FOBJ>
class SignedDistanceFieldFaceAlgoT : public SignedDistanceFieldFaceAlgo
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};



class DistanceFieldEdgeAlgo : public DynamicAlgoBase
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};

template<class FSRC, class FDST, class FOBJ>
class DistanceFieldEdgeAlgoT : public DistanceFieldEdgeAlgo
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};


class DistanceFieldNodeAlgo : public DynamicAlgoBase
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};

template<class FSRC, class FDST, class FOBJ>
class DistanceFieldNodeAlgoT : public DistanceFieldNodeAlgo
{
public:
  virtual bool DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle object);
};



template<class FSRC, class FDST, class FOBJ, class DFOBJ>
bool DistanceFieldCellAlgoT<FSRC,FDST,FOBJ,DFOBJ>::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objectfield, FieldHandle dobjectfield)
{
  FOBJ* objfield = dynamic_cast<FOBJ* >(objectfield.get_rep());
  if (objfield == 0)
  {
    pr->error("DistanceField: Object is not valid");
    return(false);
  }

  DFOBJ* dobjfield = dynamic_cast<DFOBJ* >(dobjectfield.get_rep());
  if (dobjfield == 0)
  {
    pr->error("DistanceField: Object is not valid");
    return(false);
  }

  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DistanceField: There is no input field");
    return(false);
  }

  typename FSRC::mesh_type *imesh = ifield->get_typed_mesh().get_rep();
  typename FOBJ::mesh_type *objmesh = objfield->get_typed_mesh().get_rep();
  typename DFOBJ::mesh_type *dobjmesh = dobjfield->get_typed_mesh().get_rep();

  FDST* ofield = scinew FDST(imesh);
  if (ofield == 0)
  {
    pr->error("DistanceField: Could not create output field");
    return(false);
  }

  ofield->resize_fdata();
  output = dynamic_cast<SCIRun::Field* >(ofield);  
  
  objmesh->synchronize(Mesh::LOCATE_E);
  dobjmesh->synchronize(Mesh::LOCATE_E);
  
  if (ofield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Cell::index_type cidx;
    typename DFOBJ::mesh_type::Face::index_type fidx;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p2;
      imesh->get_center(p,*(it));
      if (objmesh->locate(cidx,p))
      {
        val = 0.0; // it is inside
      }
      else
      {
        val = static_cast<typename FDST::value_type>(dobjmesh->find_closest_elem(p2,fidx,p));
      }
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Cell::index_type cidx;
    typename DFOBJ::mesh_type::Face::index_type fidx;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p2;
      imesh->get_center(p,*(it));
      if (objmesh->locate(cidx,p))
      {
        val = 0.0; // it is inside
      }
      else
      {
        val = static_cast<typename FDST::value_type>(dobjmesh->find_closest_elem(p2,fidx,p));
      }
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else
  {
    pr->error("DistanceField: Cannot add distance data to field");
    return (false);
  }
  
  return (true);  
  
  
}



template<class FSRC, class FDST, class FOBJ>
bool DistanceFieldFaceAlgoT<FSRC,FDST,FOBJ>::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objectfield)
{
  FOBJ* objfield = dynamic_cast<FOBJ* >(objectfield.get_rep());
  if (objfield == 0)
  {
    pr->error("DistanceField: Object is not valid");
    return(false);
  }

  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DistanceField: There is no input field");
    return(false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh().get_rep();
  typename FOBJ::mesh_handle_type objmesh = objfield->get_typed_mesh().get_rep();

  FDST* ofield = scinew FDST(imesh);
  if (ofield == 0)
  {
    pr->error("DistanceField: Could not create output field");
    return(false);
  }

  ofield->resize_fdata();
  output = dynamic_cast<SCIRun::Field* >(ofield);

  objmesh->synchronize(Mesh::LOCATE_E);

  if (ofield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Face::index_type fidx;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p2;
      imesh->get_center(p,*(it));
      val = static_cast<typename FDST::value_type>(objmesh->find_closest_elem(p2,fidx,p));
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Face::index_type fidx;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p2;
      imesh->get_center(p,*(it));
      val = static_cast<typename FDST::value_type>(objmesh->find_closest_elem(p2,fidx,p));
      ofield->set_value(val,*(it));
      ++it;
    }  
  }
  else
  {
    pr->error("DistanceField: Cannot add distance data to field");
    return (false);
  }
  
  return (true);
}


template<class FSRC, class FDST, class FOBJ>
bool DistanceFieldEdgeAlgoT<FSRC,FDST,FOBJ>::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objectfield)
{

  FOBJ* objfield = dynamic_cast<FOBJ* >(objectfield.get_rep());
  if (objfield == 0)
  {
    pr->error("DistanceField: Object is not valid");
    return(false);
  }

  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DistanceField: There is no input field");
    return(false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh().get_rep();
  typename FOBJ::mesh_handle_type objmesh = objfield->get_typed_mesh().get_rep();

  FDST* ofield = scinew FDST(imesh);
  if (ofield == 0)
  {
    pr->error("DistanceField: Could not create output field");
    return(false);
  }

  ofield->resize_fdata();
  output = dynamic_cast<SCIRun::Field* >(ofield);

  if (ofield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p,p1, p2;
      imesh->get_center(p,*(it));

      typename FOBJ::mesh_type::Elem::iterator oit, oit_end;
      typename FOBJ::mesh_type::Node::array_type nodes;
      double mindist = DBL_MAX;
      double dist;
      
      objmesh->begin(oit);
      objmesh->end(oit_end);
      while (oit != oit_end)
      {
        objmesh->get_nodes(nodes,*(oit));
        objmesh->get_center(p1,nodes[0]);
        objmesh->get_center(p2,nodes[1]);

        if (Dot(Vector(p-p2),Vector(p2-p1)) >= 0.0)
        {
          Vector v = Vector(p-p2);
          dist  = Dot(v,v);
        }
        else if (Dot(Vector(p-p1),Vector(p1-p2)) >= 0.0) 
        {
          Vector v = Vector(p-p1);
          dist = Dot(v,v);
        }
        else
        {
          Vector v1 = Vector(p1-p2);
          Vector v = Vector(p-p2)-v1*(Dot(Vector(p-p2),v1)/Dot(v1,v1));
          dist = Dot(v,v);
        }
        
        if (dist < mindist) mindist = dist; 
        ++oit;
      }

      val = static_cast<typename FDST::value_type>(sqrt(mindist));
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p,p1, p2;
      imesh->get_center(p,*(it));
      typename FOBJ::mesh_type::Elem::iterator oit, oit_end;
      typename FOBJ::mesh_type::Node::array_type nodes;
      double mindist = DBL_MAX;
      double dist;
      
      objmesh->begin(oit);
      objmesh->end(oit_end);
      while (oit != oit_end)
      {
        objmesh->get_nodes(nodes,*(oit));
        objmesh->get_center(p1,nodes[0]);
        objmesh->get_center(p2,nodes[1]);

        if (Dot(Vector(p-p2),Vector(p2-p1)) >= 0.0)
        {
          Vector v = Vector(p-p2);
          dist  = Dot(v,v);
        }
        else if (Dot(Vector(p-p1),Vector(p1-p2)) >= 0.0) 
        {
          Vector v = Vector(p-p1);
          dist = Dot(v,v);
        }
        else
        {
          Vector v1 = Vector(p1-p2);
          Vector v = Vector(p-p2)-v1*(Dot(Vector(p-p2),v1)/Dot(v1,v1));
          dist = Dot(v,v);
        }
        
        if (dist < mindist) mindist = dist; 
        ++oit;
      }

      val = static_cast<typename FDST::value_type>(sqrt(mindist));
      
      ofield->set_value(val,*(it));
      ++it;
    }  
  }
  else
  {
    pr->error("DistanceField: Cannot add distance data to field");
    return (false);
  }
  
  return (true);
}



template<class FSRC, class FDST, class FOBJ>
bool DistanceFieldNodeAlgoT<FSRC,FDST,FOBJ>::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objectfield)
{

  FOBJ* objfield = dynamic_cast<FOBJ* >(objectfield.get_rep());
  if (objfield == 0)
  {
    pr->error("DistanceField: Object is not valid");
    return(false);
  }

  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DistanceField: There is no input field");
    return(false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh().get_rep();
  typename FOBJ::mesh_handle_type objmesh = objfield->get_typed_mesh().get_rep();

  FDST* ofield = scinew FDST(imesh);
  if (ofield == 0)
  {
    pr->error("DistanceField: Could not create output field");
    return(false);
  }

  ofield->resize_fdata();
  output = dynamic_cast<SCIRun::Field* >(ofield);

  if (ofield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p2;
      imesh->get_center(p,*(it));
      typename FOBJ::mesh_type::Elem::iterator oit, oit_end;
      double mindist = DBL_MAX;
      double dist;
      
      objmesh->begin(oit);
      objmesh->end(oit_end);
      while (oit != oit_end)
      {
        objmesh->get_center(p2,*oit);
        Vector v = Vector(p2-p);
        dist = Dot(v,v);
        if (dist < mindist) mindist = dist;
        ++oit;
      }

      val = static_cast<typename FDST::value_type>(sqrt(mindist));
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val;
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p2;
      imesh->get_center(p,*(it));
      typename FOBJ::mesh_type::Elem::iterator oit, oit_end;
      double mindist = DBL_MAX;
      double dist;
      
      objmesh->begin(oit);
      objmesh->end(oit_end);
      while (oit != oit_end)
      {
        objmesh->get_center(p2,*oit);
        Vector v = Vector(p2-p);
        dist = Dot(v,v);
        if (dist < mindist) mindist = dist;
        ++oit;
      }

      val = static_cast<typename FDST::value_type>(sqrt(mindist));
      ofield->set_value(val,*(it));
      ++it;
    }  
  }
  else
  {
    pr->error("DistanceField: Cannot add distance data to field");
    return (false);
  }
  
  return (true);
}


template<class FSRC, class FDST, class FOBJ>
bool SignedDistanceFieldFaceAlgoT<FSRC,FDST,FOBJ>::DistanceField(ProgressReporter *pr, FieldHandle input, FieldHandle& output, FieldHandle objectfield)
{
  FOBJ* objfield = dynamic_cast<FOBJ* >(objectfield.get_rep());
  if (objfield == 0)
  {
    pr->error("DistanceField: Object is not valid");
    return(false);
  }

  FSRC* ifield = dynamic_cast<FSRC* >(input.get_rep());
  if (ifield == 0)
  {
    pr->error("DistanceField: There is no input field");
    return(false);
  }

  typename FSRC::mesh_handle_type imesh = ifield->get_typed_mesh().get_rep();
  typename FOBJ::mesh_handle_type objmesh = objfield->get_typed_mesh().get_rep();

  FDST* ofield = scinew FDST(imesh);
  if (ofield == 0)
  {
    pr->error("DistanceField: Could not create output field");
    return(false);
  }

  ofield->resize_fdata();
  output = dynamic_cast<SCIRun::Field* >(ofield);

  objmesh->synchronize(Mesh::LOCATE_E);

  if (ofield->basis_order() == 0)
  {
    typename FSRC::mesh_type::Elem::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Face::index_type fidx, fidx_n;
    typename FOBJ::mesh_type::Node::array_type nodes;
    typename FOBJ::mesh_type::Edge::array_type edges;
    Vector n, k;
    Point n0,n1,n2;
    
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p1, p2;
      imesh->get_center(p,*(it));
      val = static_cast<typename FDST::value_type>(objmesh->find_closest_elem(p2,fidx,p));
      objmesh->get_nodes(nodes,fidx);  
      objmesh->get_center(n0,nodes[0]);
      objmesh->get_center(n1,nodes[1]);
      objmesh->get_center(n2,nodes[2]);
      n = Cross(Vector(n1-n0),Vector(n2-n1));
      k = Vector(p-p2);
      k.normalize();
      double angle = Dot(n,k);
      if (angle < 1e-6)
      {
        val = -val;
      }
      else if (angle > 1e-6)
      {
      }
      else
      {
        // trouble
        if (val != 0)
        {
           objmesh->get_edges(edges,fidx);
           double mindist = DBL_MAX;
           double dist;
           int edgeidx = 0;
           for (size_t r=0; r<edges.size();r++)
           {
             objmesh->get_nodes(nodes,edges[r]);
             objmesh->get_center(p1,nodes[0]);
             objmesh->get_center(p2,nodes[1]);

            if (Dot(Vector(p-p2),Vector(p2-p1)) >= 0.0)
            {
              Vector v = Vector(p-p2);
              dist  = Dot(v,v);
            }
            else if (Dot(Vector(p-p1),Vector(p1-p2)) >= 0.0) 
            {
              Vector v = Vector(p-p1);
              dist = Dot(v,v);
            }
            else
            {
              Vector v1 = Vector(p1-p2);
              Vector v = Vector(p-p2)-v1*(Dot(Vector(p-p2),v1)/Dot(v1,v1));
              dist = Dot(v,v);
            }
            
            if (dist < mindist) { mindist = dist; edgeidx = r;}
          }
          objmesh->get_neighbor(fidx_n,fidx,edges[edgeidx]);
          objmesh->get_nodes(nodes,fidx);  
          objmesh->get_center(n0,nodes[0]);
          objmesh->get_center(n1,nodes[1]);
          objmesh->get_center(n2,nodes[2]);
          n = Cross(Vector(n1-n0),Vector(n2-n1));
          k = Vector(p-p2);
          k.normalize();
          angle = Dot(n,k);
          if (angle < 0) val = -(val);
        }
      }
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else if (ofield->basis_order() == 1)
  {
    typename FSRC::mesh_type::Node::iterator it, it_end;
    typename FDST::value_type val;
    typename FOBJ::mesh_type::Face::index_type fidx, fidx_n;
    typename FOBJ::mesh_type::Node::array_type nodes;
    typename FOBJ::mesh_type::Edge::array_type edges;
    Vector n, k;
    Point n0,n1,n2;
    
    imesh->begin(it); 
    imesh->end(it_end);

    while (it != it_end)
    {
      Point p, p1, p2;
      imesh->get_center(p,*(it));
      val = static_cast<typename FDST::value_type>(objmesh->find_closest_elem(p2,fidx,p));
      objmesh->get_nodes(nodes,fidx);  
      objmesh->get_center(n0,nodes[0]);
      objmesh->get_center(n1,nodes[1]);
      objmesh->get_center(n2,nodes[2]);
      n = Cross(Vector(n1-n0),Vector(n2-n1));
      k = Vector(p-p2);
      k.normalize();
      double angle = Dot(n,k);
      if (angle < 1e-6)
      {
        val = -val;
      }
      else if (angle > 1e-6)
      {
      }
      else
      {
        // trouble
        if (val != 0)
        {
           objmesh->get_edges(edges,fidx);
           double mindist = DBL_MAX;
           double dist;
           int edgeidx = 0;
           for (size_t r=0; r<edges.size();r++)
           {
             objmesh->get_nodes(nodes,edges[r]);
             objmesh->get_center(p1,nodes[0]);
             objmesh->get_center(p2,nodes[1]);

            if (Dot(Vector(p-p2),Vector(p2-p1)) >= 0.0)
            {
              Vector v = Vector(p-p2);
              dist  = Dot(v,v);
            }
            else if (Dot(Vector(p-p1),Vector(p1-p2)) >= 0.0) 
            {
              Vector v = Vector(p-p1);
              dist = Dot(v,v);
            }
            else
            {
              Vector v1 = Vector(p1-p2);
              Vector v = Vector(p-p2)-v1*(Dot(Vector(p-p2),v1)/Dot(v1,v1));
              dist = Dot(v,v);
            }
            
            if (dist < mindist) { mindist = dist; edgeidx = r;}
          }
          objmesh->get_neighbor(fidx_n,fidx,edges[edgeidx]);
          objmesh->get_nodes(nodes,fidx);  
          objmesh->get_center(n0,nodes[0]);
          objmesh->get_center(n1,nodes[1]);
          objmesh->get_center(n2,nodes[2]);
          n = Cross(Vector(n1-n0),Vector(n2-n1));
          k = Vector(p-p2);
          k.normalize();
          angle = Dot(n,k);
          if (angle < 0) val = -(val);
        }
      }
      ofield->set_value(val,*(it));
      ++it;
    }
  }
  else
  {
    pr->error("DistanceField: Cannot add distance data to field");
    return (false);
  }
  
  return (true);
}

} // end namespace SCIRunAlgo

#endif
