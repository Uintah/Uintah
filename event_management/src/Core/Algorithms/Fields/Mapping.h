/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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

#ifndef CORE_ALGORITHMS_FIELDS_MAPPING_H
#define CORE_ALGORITHMS_FIELDS_MAPPING_H 1

#include <Core/Algorithms/Util/DynamicAlgo.h>
#include <sci_hash_map.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <algorithm>
#include <sgi_stl_warnings_on.h>

namespace SCIRunAlgo {

using namespace SCIRun;

// MappingMethod:
//  How do we select data
//  ClosestNodalData = Find the closest data containing element or node
//  ClosestInterpolatedData = Find the closest interpolated data point. Inside the volume it picks the value
//                            the interpolation model predicts and outside it makes a shortest projection
//  InterpolatedData = Uses interpolated data using the interpolation model whereever possible and assumes no
//                     value outside the source field


class NodalMappingAlgo : public DynamicAlgoBase
{
public:

  virtual bool NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output 
                       std::string mappingmethod);
};

class ModalMappingAlgo : public DynamicAlgoBase
{
public:

  virtual bool ModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter );
};




template <class MAPPING, class FSRC, class FDST, class FOUT>
class NodalMappingAlgoT : public NodalMappingAlgo
{
public:
  virtual bool NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output 
                       std::string mappingmethod);
                       
  class IData
  {
    public:
      ProgressReporter*         pr;
      FSRC*                     ifield;
      FOUT*                     ofield;
      typename FSRC::mesh_type* imesh;
      typename FOUT::mesh_type* omesh;
      int                       numproc;
  };
  
  void parallel(int procnum,IData* inputdata); 
};



template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
class ModalMappingAlgoT : public ModalMappingAlgo
{
public:
  virtual bool ModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter );
                       
  class IData
  {
    public:
      ProgressReporter*         pr;
      FSRC*                     ifield;
      FOUT*                     ofield;
      typename FSRC::mesh_type* imesh;
      typename FOUT::mesh_type* omesh;
      int                       numproc;
      std::string               integrationfilter;
      bool                      retval;
  };
  
  void parallel(int procnum,IData* inputdata); 
};





template <class MAPPING, class FSRC, class FDST, class FOUT>
bool NodalMappingAlgoT<MAPPING,FSRC,FDST,FOUT>::NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,FieldHandle dst, FieldHandle& output 
                       std::string mappingmethod) 
{
  FSRC* ifield = dynamic_cast<FSRC*>(src.get_rep());
  if (ifield == 0)
  {
    pr->error("NodalMapping: No input source field was given");
    return (false);
  }

  typename FSRC::mesh_type* imesh = dynamic_cast<typename FSRC::mesh_type*>(src->mesh().get_rep());
  if (imesh == 0)
  {
    pr->error("NodalMapping: No mesh is associated with input source field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("NodalMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("NodalMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("NodalMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(input.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.ifield = ifield;
  IData.ofield = ofield;
  IData.imesh  = imesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&NodalMappingAlgoT<MAPPING,FSRC,FDST,FOUT>::parallel,np,&IData);
    
  return (true);
}


template <class MAPPING, class FSRC, class FDST, class FOUT>
void NodalMappingAlgoT<MAPPING,FSRC,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Node::iterator it, eit;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val;
  
  int numproc = idata->numproc;
  
  omesh->begin(it);
  omesh->end(eit);
  
  MAPPING mapping(idata->ifield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  Point point;
  
  while (it != eit)
  {
    omesh->get_center(point,*it);
    mapping.get_data(point,val);
    omesh->set_value(val,*it);
  
    for (int p =0; p < numproc; p++) if (it != eit) ++it;  
  }
}





template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
bool ModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,FieldHandle dst, FieldHandle& output 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter) 
{
  FSRC* ifield = dynamic_cast<FSRC*>(src.get_rep());
  if (ifield == 0)
  {
    pr->error("NodalMapping: No input source field was given");
    return (false);
  }

  typename FSRC::mesh_type* imesh = dynamic_cast<typename FSRC::mesh_type*>(src->mesh().get_rep());
  if (imesh == 0)
  {
    pr->error("NodalMapping: No mesh is associated with input source field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("NodalMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("NodalMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("NodalMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(input.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.ifield = ifield;
  IData.ofield = ofield;
  IData.imesh  = imesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  IData.integrationfilter = integrationfilter;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&ModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel,np,&IData);
    
  return (IData->retval);
}



template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
void ModalMappingAlgoT<MAPPING,FSRC,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Node::iterator it, eit;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val, val2;
  
  int numproc = idata->numproc;
  
  omesh->begin(it);
  omesh->end(eit);
  
  MAPPING mapping(idata->ifield);
  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::string filter = idata->integrationfilterl
  
  if ((filter == "median")||(filter == "Median"))
  {
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        mapping.get_data(points[p],valarray[p]);
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "minimum")||(filter == "Minimum"))
  {
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val = 0.0;
      typename FOUT::value_type tval = 0.0;

      if (points.size() > 0)
      {
        mapping.get_data(points[0],val);
        for (size_t p = 1; p < points.size(); p++)
        {
          mapping.get_data(points[p],tval);
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "maximum")||(filter == "Maximum"))
  {
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val = 0.0;
      typename FOUT::value_type tval = 0.0;

      if (points.size() > 0)
      {
        mapping.get_data(points[0],val);
        for (size_t p = 1; p < points.size(); p++)
        {
          mapping.get_data(points[p],tval);
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "mostcommon")||(filter == "Mostcommon"))
  {
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        mapping.get_data(points[p],valarray[p]);
      }
      sort(valarray.begin(),valarray.end());
       
      typename FOUT::value_type rval = 0;
      typename FOUT::value_type val = 0;
      int rnum = 0;
      
      int p = 0;
      int n = 0;
      
      while (1)
      {
        if (p < valarray.size())
        {
          n = 1;
          val = valarray[p];
        } 
        p++;
        while ( p < valarray.size() && valarray[p] = val) { n++; p++; }
        
        if (n > rnum) { rnum = n; rval = val;}
      }
            
      ofield->set_value(rval,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate")||(filter == "Interpolate")||(filter == "interpolate"))
  {
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        mapping.get_data(points[p],val2);
        val += val2*weights[p];
      }
      omesh->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        mapping.get_data(points[p],val2);
        val += val2 * (1.0/points.size());
      }
      omesh->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else
  {
    if (numproc == 0)
    {
      idata->pr->error("ModalMapping: Filter method is unknown");
      idata->retval = false;
      return;
    }
  }
  
  if (numproc == 0) idata->retval = true;
}













// Classes for finding values in a mesh

template <class FIELD>
class ClosestNodalData {
  public:
    ClosestNodalData(FIELD* field);
    ~ClosestNodalData();
    
    bool get_data(Point& p, typename FIELD::value_type& val);

  private:
    // Store these so we do need to reserve memory each time
    FIELD*  field_;
    typename FIELD::mesh_type mesh_;

    typename FSRC::mesh_type::Elem::index_type idx_;
    typename FSRC::mesh_type::Node::array_type nodes_;    
    
    typename FSRC::mesh_type::Node::iterator it_;
    typename FSRC::mesh_type::Node::iterator eit_;
    typename FSRC::mesh_type::Node::index_type minidx_;
};


template <class FIELD>
ClosestNodalData<FIELD>::ClosestNodalData(FIELD* field)
{
  field_ = field;
  mesh_ = field->get_typed_mesh().get_rep();
  mesh_->synchronize(Mesh::LOCATE_E);
}

template <class FIELD>
ClosestNodalData<FIELD>::~ClosestNodalData()
{
}

template <class FIELD>
bool ClosestNodalData<FIELD>::get_data(Point& p, typename FIELD::value_type& val)
{
  Point p2;
  double dist = MAX_DBL;
  double distt;
  
  if (mesh_->locate(idx_,p))
  {
    mesh_->get_nodes(nodes_,idx_);
    
    int minidx = 0;
    for (int r =0; r < nodes_.size(); r++)
    {
      mesh_->get_center(p2,nodes_[r]);  
      distt = Vector(p2-p).length2(); 
      if (distt < dist)
      {
        dist = distt;
        minidx = r;
      }
    }
    
    field_->value(data,nodes_[minidx]);
    return  (true);
  }

  mesh_->begin(it_);
  mesh_->end(eit_);
  while (it_ != eit_)
  {
    Point c;
    mesh_->get_center(c, *it_);
    const double distt = (p - c).length2();
    if (distt < dist)
    {
      dist = distt;
      minidx = *it_;
    }
    ++it_;
  }
 
  field_->value(data,minidx);
  return (true);
}




template <FIELD>
class ClosestModalData {
  public:
    ClosestModalData(FIELD* field);
    ~ClosestModalData();
    
    bool get_data(Point& p, typename FIELD::value_type& val);

  private:
    // Store these so we do need to reserve memory each time
    FIELD*  field_;
    typename FIELD::mesh_type mesh_;

    typename FSRC::mesh_type::Elem::index_type idx_;
    
    typename FSRC::mesh_type::Elem::iterator it_;
    typename FSRC::mesh_type::Elem::iterator eit_;
    typename FSRC::mesh_type::Elem::index_type minidx_;

};


template <class FIELD>
ClosestModalData<FIELD>::ClosestModalData(FIELD* field)
{
  field_ = field;
  mesh_ = field->get_typed_mesh().get_rep();
  mesh_->synchronize(Mesh::LOCATE_E);
}

template <class FIELD>
ClosestModalData<FIELD>::~ClosestModalData()
{
}

template <class FIELD>
bool ClosestModalData<FIELD>::get_data(Point& p, typename FIELD::value_type& val)
{
  Point p2;
  double dist = MAX_DBL;
  double distt;
  
  if (mesh_->locate(idx_,p))
  {    
    field_->value(data,idx_);
    return  (true);
  }

  mesh_->begin(it_);
  mesh_->end(eit_);
  while (it_ != eit_)
  {
    Point c;
    mesh_->get_center(c, *it_);
    const double distt = (p - c).length2();
    if (distt < dist)
    {
      dist = distt;
      minidx = *it_;
    }
    ++it_;
  }
 
  field_->value(data,minidx);
  return (true);
}


// InterpolatedData class which gets the
// interpolated value from a mesh

template <class FIELD>
class InterpolatedData {
  public:
    InterpolatedData(FIELD* field);
    ~InterpolatedData();
    
    bool get_data(Point& p, typename FIELD::value_type& val);
    
  private:
    FIELD*              field_;
    FIELD::mesh_type*   mesh_;
             
    std::vector<double> coords_;
    typename FIELD::mesh_type::Elem::index_type idx_;
};


template <class FIELD>
InterpolatedData<FIELD>::InterpolatedData<FIELD>(FIELD* field)
{
  field_ = field;
  mesh_ = field->get_typed_mesh().get_rep();
  mesh_->synchronize(Mesh::LOCATE_E);
}

template <class FIELD>
InterpolatedData<FIELD>::~InterpolatedData<FIELD>()
{
}

template <class FIELD>
bool InterpolatedData<FIELD>::get_data(Point& p, typename FIELD::value_type& data)
{
  
  if (mesh_->locate(idx_,p))
  {
    field_->interpolate(data,coord_,idx_);
    return (true);
  }
  
  return (false);
}




/*
template <FIELD,DFIELD>
class ClosestInterpolatedCellData {
  public:
    ClosestInterpolatedCellData(FIELD* field);
    ~ClosestInterpolatedCellData();
    
    bool get_data(Point& p, typename FIELD::value_type& val);
    
  private:
    FIELD*              field_;
    DFIELD*             dfield_;
    FIELD::mesh_type*   mesh_;

    FieldHandle         handle_;
               
    std::vector<double> coords_;
};


template <class FIELD, class DFIELD>
bool ClosestInterpolatedCellData<FIELD,DFIELD>::ClosestInterpolatedCellData<FIELD>(FIELD* field)
{
  field_ = field;
  mesh_ = field->get_typed_mesh().get_rep();
  mesh_->synchronize(Mesh::LOCATE_E);
}



template <class FIELD, class DFIELD>
bool ClosestInterpolatedData<FIELD,DFIELD>::get_data(Point& p, typename FIELD::value_type& data)
{
  typename FSRC::mesh_type::Elem::index_type idx;
  
  if (mesh_->locate(idx,p))
  {
    std::vector<double> coords;
    field_->interpolate(data,coord_,idx);
    return (true);
  }
  
  
}
*/



}

#endif

