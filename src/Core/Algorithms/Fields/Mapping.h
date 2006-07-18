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

#include <Core/Basis/Locate.h>
#include <Core/Algorithms/Util/DynamicAlgo.h>

#include <float.h>

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

// Mapping Method to nodes (nodal values)

class NodalMappingAlgo : public DynamicAlgoBase
{
public:

  virtual bool NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod, double def_value);
};


// IntegrationMethod:
//  Gaussian1 = Use 1st order Gaussian weights and nodes for integration
//  Gaussian2 = Use 2nd order Gaussian weights and nodes for integration
//  Gaussian3 = Use 3rd order Gaussian weights and nodes for integration

// Integration Filter:
//  Average =  take average value over integration nodes but disregard weights
//  Integrate = sum values over integration nodes using gaussian weights
//  Minimum = find minimum value using integration nodes
//  Maximum = find maximum value using integration nodes
//  Median  = find median value using integration nodes
//  MostCommon = find most common value among integration nodes


// Mapping Method to elements (Modal values)

class ModalMappingAlgo : public DynamicAlgoBase
{
public:

  virtual bool ModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       double def_value );
};


class GradientModalMappingAlgo : public DynamicAlgoBase
{
public:

  virtual bool GradientModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool calcnorm);
};

// Templated class for Nodal Mapping

template <class MAPPING, class FSRC, class FDST, class FOUT>
class NodalMappingAlgoT : public NodalMappingAlgo
{
public:
  virtual bool NodalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       double def_value);
  
  // Input data for parallel algorithm
  class IData
  {
    public:
      ProgressReporter*         pr;
      FSRC*                     ifield;
      FOUT*                     ofield;
      typename FSRC::mesh_type* imesh;
      typename FOUT::mesh_type* omesh;
      int                       numproc;  // number of processes
      bool                      retval;   // return value
      double                    def_value;
  };
  
  // Parallel implementation
  void parallel(int procnum,IData* inputdata); 
};


// Modal version
template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
class ModalMappingAlgoT : public ModalMappingAlgo
{
public:
  virtual bool ModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       double def_value);
                       
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
      double                    def_value;
  };
  
  void parallel(int procnum,IData* inputdata); 
};


// Modal version
template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
class GradientModalMappingAlgoT : public GradientModalMappingAlgo
{
public:
  virtual bool GradientModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool calnorm);
                       
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

template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
class GradientModalMappingNormAlgoT : public GradientModalMappingAlgo
{
public:
  virtual bool GradientModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,
                       FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool calnorm);
                       
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
                       int numproc, FieldHandle src,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod, double def_value) 
{
  // Some sanity checks, in order not to crash SCIRun when some is generating nonsense input
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

  // Creating output mesh and field
  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("NodalMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  // Make sure it inherits all the properties
  output->copy_properties(dst.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.ifield = ifield;
  IData.ofield = ofield;
  IData.imesh  = imesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  IData.retval = true;
  IData.def_value = def_value;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&NodalMappingAlgoT<MAPPING,FSRC,FDST,FOUT>::parallel,np,&IData);
    
  return (IData.retval);
}


template <class MAPPING, class FSRC, class FDST, class FOUT>
void NodalMappingAlgoT<MAPPING,FSRC,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Node::iterator it, eit;
  typename FOUT::mesh_type* omesh = idata->omesh;
  FOUT* ofield = idata->ofield;
  typename FOUT::value_type val;
  
  int numproc = idata->numproc;
  
  // loop over all the output nodes
  omesh->begin(it);
  omesh->end(eit);
  
  // Define class that defines how we find data from the source mesh
  MAPPING mapping(idata->ifield);
  
  // Make sure we start with the proper node
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  Point point;
  
  while (it != eit)
  {
    // Find the destination location
    omesh->get_center(point,*it);
    // Find the value associated with that location
    // This is the operation that should take most time
    // Hence we use it as a template so it can be compiled fully inline
    // without having to generate a lot of code
    if(!(mapping.get_data(point,val))) val = idata->def_value;
    // Set the value
    ofield->set_value(val,*it);
  
    // Skip to the next one, but disregard the nodes the other 
    // processes are working on.
    for (int p =0; p < numproc; p++) if (it != eit) ++it;  
  }
}



// Modal version

template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
bool ModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::ModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       double def_value) 
{
  FSRC* ifield = dynamic_cast<FSRC*>(src.get_rep());
  if (ifield == 0)
  {
    pr->error("ModalMapping: No input source field was given");
    return (false);
  }

  typename FSRC::mesh_type* imesh = dynamic_cast<typename FSRC::mesh_type*>(src->mesh().get_rep());
  if (imesh == 0)
  {
    pr->error("ModalMapping: No mesh is associated with input source field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("ModalMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("ModalMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("ModalMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(dst.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.ifield = ifield;
  IData.ofield = ofield;
  IData.imesh  = imesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  IData.integrationfilter = integrationfilter;
  IData.def_value = def_value;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&ModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel,np,&IData);
    
  return (IData.retval);
}



template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
void ModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Elem::iterator it, eit;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val, val2;
  FOUT* ofield = idata->ofield;
  
  int numproc = idata->numproc;
  
  omesh->begin(it);
  omesh->end(eit);
  
  MAPPING mapping(idata->ifield);
  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::string filter = idata->integrationfilter;
  
  // Determine the filter and loop over nodes
  if ((filter == "median")||(filter == "Median"))
  {
    // median filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if(!(mapping.get_data(points[p],valarray[p]))) valarray[p] = idata->def_value;
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "minimum")||(filter == "Minimum"))
  {
    // minimum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val = 0.0;
      typename FOUT::value_type tval = 0.0;

      if (points.size() > 0)
      {
        if(!(mapping.get_data(points[0],val))) val = idata->def_value;
        for (size_t p = 1; p < points.size(); p++)
        {
          if(!(mapping.get_data(points[p],tval))) tval = idata->def_value;
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "maximum")||(filter == "Maximum"))
  {
    // maximum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val = 0.0;
      typename FOUT::value_type tval = 0.0;

      if (points.size() > 0)
      {
        if (!(mapping.get_data(points[0],val))) val = idata->def_value;
        for (size_t p = 1; p < points.size(); p++)
        {
          if (!(mapping.get_data(points[p],tval))) val = idata->def_value;
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "mostcommon")||(filter == "Mostcommon")||(filter == "MostCommon"))
  {
    // Filter designed for segmentations where one wants the most common element to be the
    // sampled element
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if(!(mapping.get_data(points[p],valarray[p]))) valarray[p] = idata->def_value;
      }
      sort(valarray.begin(),valarray.end());
       
      typename FOUT::value_type rval = 0;
      typename FOUT::value_type val = 0;
      int rnum = 0;
      
      int p = 0;
      int n = 0;
      
      while (p < valarray.size())
      {
        n = 1;
        val = valarray[p];

        p++;
        while ( p < valarray.size() && valarray[p] == val) { n++; p++; }
        
        if (n > rnum) { rnum = n; rval = val;}
      }
            
      ofield->set_value(rval,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_iweights(*it,points,weights);

      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_data(points[p],val2))) val2 = idata->def_value;
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "weightedaverage")||(filter == "WeightedAverage"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);

      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_data(points[p],val2))) val2 = idata->def_value;
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_data(points[p],val2))) val2 = idata->def_value;
        val += val2 * (1.0/points.size());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "sum")||(filter == "Sum"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if (!(mapping.get_data(points[p],val2))) val2 = idata->def_value;
        val += val2;;
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else
  {
    if (numproc == 0)
    {
      idata->pr->error("ModalMapping: Filter method is unknown");
      idata->retval = false;
    }
    return;
  }
  
  if (numproc == 0) idata->retval = true;
}



// GradientModal version

template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
bool GradientModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::GradientModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter, bool calcnorm) 
{
  FSRC* ifield = dynamic_cast<FSRC*>(src.get_rep());
  if (ifield == 0)
  {
    pr->error("GradientModalMapping: No input source field was given");
    return (false);
  }

  typename FSRC::mesh_type* imesh = dynamic_cast<typename FSRC::mesh_type*>(src->mesh().get_rep());
  if (imesh == 0)
  {
    pr->error("GradientModalMapping: No mesh is associated with input source field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("GradientModalMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("GradientModalMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("GradientModalMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(dst.get_rep());

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
   
  Thread::parallel(this,&GradientModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel,np,&IData);
        
  return (IData.retval);
}



template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
void GradientModalMappingAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Elem::iterator it, eit;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val, val2;
  FOUT* ofield = idata->ofield;
  
  int numproc = idata->numproc;
  
  omesh->begin(it);
  omesh->end(eit);
  
  MAPPING mapping(idata->ifield);
  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::string filter = idata->integrationfilter;
  
  // Determine the filter and loop over nodes
  if ((filter == "median")||(filter == "Median"))
  {
    // median filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],valarray[p]))) valarray[p] = 0;
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "minimum")||(filter == "Minimum"))
  {
    // minimum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val; val = 0;
      typename FOUT::value_type tval; tval = 0;

      if (points.size() > 0)
      {
        if(!(mapping.get_gradient(points[0],val))) val = 0;
        for (size_t p = 1; p < points.size(); p++)
        {
          if(!(mapping.get_gradient(points[p],tval))) tval = 0;
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "maximum")||(filter == "Maximum"))
  {
    // maximum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val; val = 0;
      typename FOUT::value_type tval; tval = 0;

      if (points.size() > 0)
      {
        if (!(mapping.get_gradient(points[0],val))) val = 0;
        for (size_t p = 1; p < points.size(); p++)
        {
          if (!(mapping.get_gradient(points[p],tval))) tval = 0;
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "mostcommon")||(filter == "Mostcommon")||(filter == "MostCommon"))
  {
    // Filter designed for segmentations where one wants the most common element to be the
    // sampled element
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],valarray[p]))) valarray[p] = 0;
      }
      sort(valarray.begin(),valarray.end());
       
      typename FOUT::value_type rval; rval = 0;
      typename FOUT::value_type val; val = 0;
      int rnum = 0;
      
      int p = 0;
      int n = 0;
      
      while (p < valarray.size())
      {
        n = 1;
        val = valarray[p];

        p++;
        while ( p < valarray.size() && valarray[p] == val) { n++; p++; }
        
        if (n > rnum) { rnum = n; rval = val;}
      }
            
      ofield->set_value(rval,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_iweights(*it,points,weights);

      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "weightedaverage")||(filter == "WeightedAverage"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);

      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);      
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += val2 * (1.0/points.size());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "sum")||(filter == "Sum"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if (!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += val2;
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else
  {
    if (numproc == 0)
    {
      idata->pr->error("GradientModalMapping: Filter method is unknown");
      idata->retval = false;
    }
    return;
  }
  
  if (numproc == 0) idata->retval = true;
}





template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
bool GradientModalMappingNormAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::GradientModalMapping(ProgressReporter *pr,
                       int numproc, FieldHandle src,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter, bool calcnorm) 
{
  FSRC* ifield = dynamic_cast<FSRC*>(src.get_rep());
  if (ifield == 0)
  {
    pr->error("GradientModalMapping: No input source field was given");
    return (false);
  }

  typename FSRC::mesh_type* imesh = dynamic_cast<typename FSRC::mesh_type*>(src->mesh().get_rep());
  if (imesh == 0)
  {
    pr->error("GradientModalMapping: No mesh is associated with input source field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("GradientModalMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("GradientModalMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("GradientModalMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(dst.get_rep());

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
   
  Thread::parallel(this,&GradientModalMappingNormAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel,np,&IData);
        
  return (IData.retval);
}



template <class MAPPING, class INTEGRATOR, class FSRC, class FDST, class FOUT>
void GradientModalMappingNormAlgoT<MAPPING,INTEGRATOR,FSRC,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Elem::iterator it, eit;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val; 
  Vector val2;
  FOUT* ofield = idata->ofield;
  
  int numproc = idata->numproc;
  
  omesh->begin(it);
  omesh->end(eit);
  
  MAPPING mapping(idata->ifield);
  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::string filter = idata->integrationfilter;
  
  // Determine the filter and loop over nodes
  if ((filter == "median")||(filter == "Median"))
  {
    // median filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        valarray[p] = static_cast<typename FOUT::value_type>(val2.length());
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "minimum")||(filter == "Minimum"))
  {
    // minimum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val; val = 0;
      typename FOUT::value_type tval; tval = 0;

      if (points.size() > 0)
      {
        if(!(mapping.get_gradient(points[0],val2))) val2 = 0;
        val = static_cast<typename FOUT::value_type>(val2.length());
        for (size_t p = 1; p < points.size(); p++)
        {
          if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
          tval = static_cast<typename FOUT::value_type>(val2.length());
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "maximum")||(filter == "Maximum"))
  {
    // maximum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      typename FOUT::value_type val; val = 0;
      typename FOUT::value_type tval; tval = 0;

      if (points.size() > 0)
      {
        if (!(mapping.get_gradient(points[0],val2))) val2 = 0;
        val = static_cast<typename FOUT::value_type>(val2.length());
        for (size_t p = 1; p < points.size(); p++)
        {
          if (!(mapping.get_gradient(points[p],val2))) val2 = 0;
          val = static_cast<typename FOUT::value_type>(val2.length());
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "mostcommon")||(filter == "Mostcommon")||(filter == "MostCommon"))
  {
    // Filter designed for segmentations where one wants the most common element to be the
    // sampled element
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      std::vector<typename FOUT::value_type> valarray(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        valarray[p] = static_cast<typename FOUT::value_type>(val2.length());
      }
      sort(valarray.begin(),valarray.end());
       
      typename FOUT::value_type rval; rval = 0;
      typename FOUT::value_type val; val = 0;
      int rnum = 0;
      
      int p = 0;
      int n = 0;
      
      while (p < valarray.size())
      {
        n = 1;
        val = valarray[p];

        p++;
        while ( p < valarray.size() && valarray[p] == val) { n++; p++; }
        
        if (n > rnum) { rnum = n; rval = val;}
      }
            
      ofield->set_value(rval,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_iweights(*it,points,weights);

      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += static_cast<typename FOUT::value_type>(val2.length())*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "weightedaverage")||(filter == "WeightedAverage"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);

      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += static_cast<typename FOUT::value_type>(val2.length())*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);      
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if(!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += static_cast<typename FOUT::value_type>(val2.length()) * (1.0/points.size());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else if ((filter == "sum")||(filter == "Sum"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0.0;
      for (int p=0; p<points.size(); p++)
      {
        if (!(mapping.get_gradient(points[p],val2))) val2 = 0;
        val += static_cast<typename FOUT::value_type>(val2.length());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
    }
  }
  else
  {
    if (numproc == 0)
    {
      idata->pr->error("GradientModalMapping: Filter method is unknown");
      idata->retval = false;
    }
    return;
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

    typename FIELD::mesh_type::Elem::index_type idx_;
    typename FIELD::mesh_type::Node::array_type nodes_;    
    
    typename FIELD::mesh_type::Node::iterator it_;
    typename FIELD::mesh_type::Node::iterator eit_;
    typename FIELD::mesh_type::Node::index_type minidx_;
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
  double dist = DBL_MAX;
  double distt;
  int minidx = 0;
  
  if (mesh_->locate(idx_,p))
  {
    mesh_->get_nodes(nodes_,idx_);
    
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
    
    field_->value(val,nodes_[minidx]);
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
 
  field_->value(val,minidx);
  return (true);
}




template <class FIELD>
class ClosestModalData {
  public:
    ClosestModalData(FIELD* field);
    ~ClosestModalData();
    
    bool get_data(Point& p, typename FIELD::value_type& val);

  private:
    // Store these so we do need to reserve memory each time
    FIELD*  field_;
    typename FIELD::mesh_type* mesh_;

    typename FIELD::mesh_type::Elem::index_type idx_;
    
    typename FIELD::mesh_type::Elem::iterator it_;
    typename FIELD::mesh_type::Elem::iterator eit_;
    typename FIELD::mesh_type::Elem::index_type minidx_;

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
  double dist = DBL_MAX;
  double distt;
  
  if (mesh_->locate(idx_,p))
  {    
    field_->value(val,idx_);
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
      minidx_ = *it_;
    }
    ++it_;
  }
 
  field_->value(val,minidx_);
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
    typename FIELD::mesh_type*   mesh_;
             
    std::vector<double> coords_;
    typename FIELD::mesh_type::Elem::index_type idx_;
};


template <class FIELD>
InterpolatedData<FIELD>::InterpolatedData(FIELD* field)
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
    mesh_->get_coords(coords_,p,idx_);
    field_->interpolate(data,coords_,idx_);
    return (true);
  }
  
  return (false);
}


template <class FIELD>
class InterpolatedGradient {
  public:
    InterpolatedGradient(FIELD* field);
    ~InterpolatedGradient();
    
    bool get_gradient(Point& p, Vector& val);
    
  private:
    FIELD*              field_;
    typename FIELD::mesh_type*   mesh_;
             
    std::vector<double> coords_;
    std::vector<typename FIELD::value_type> grad_;
    typename FIELD::mesh_type::Elem::index_type idx_;
};


template <class FIELD>
InterpolatedGradient<FIELD>::InterpolatedGradient(FIELD* field)
{
  field_ = field;
  mesh_ = field->get_typed_mesh().get_rep();
  mesh_->synchronize(Mesh::LOCATE_E);
  grad_.resize(3);
}

template <class FIELD>
InterpolatedGradient<FIELD>::~InterpolatedGradient<FIELD>()
{
}

template <class FIELD>
bool InterpolatedGradient<FIELD>::get_gradient(Point& p, Vector& data)
{
  
  if (mesh_->locate(idx_,p))
  {
    mesh_->get_coords(coords_,p,idx_);
    field_->gradient(grad_,coords_,idx_);
    data[0] = static_cast<double>(grad_[0]);
    data[1] = static_cast<double>(grad_[1]);
    data[2] = static_cast<double>(grad_[2]);
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


// Integration classes

template <class GAUSSIAN, class FIELD >
class GaussianIntegration 
{
  public:

    GaussianIntegration(FIELD* field)
    {
      field_ = field;
      if (field_)
      {
        mesh_  = field->get_typed_mesh().get_rep();
        basis_ = mesh_->get_basis();
        
        coords_.resize(gauss_.GaussianNum);
        weights_.resize(gauss_.GaussianNum);
        for (int p=0; p<gauss_.GaussianNum; p++)
        {
          for (int q=0; q<basis_.domain_dimension(); q++)
            coords_[p].push_back(gauss_.GaussianPoints[p][q]);
          weights_[p] = gauss_.GaussianWeights[p];
        }
        vol_ = basis_.volume();
        dim_ = basis_.domain_dimension();
      }
    }

    void get_nodes_and_weights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<double>& gweights)
    {    
      gpoints.resize(gauss_.GaussianNum);
      gweights.resize(gauss_.GaussianNum);
      
      for (int k=0; k < coords_.size(); k++)
      {
        mesh_->interpolate(gpoints[k],coords_[k],idx);
        gweights[k] = weights_[k];
      }
    }
        
    void get_nodes_and_iweights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<double>& gweights)
    {    
      
      gpoints.resize(gauss_.GaussianNum);
      gweights.resize(gauss_.GaussianNum);
      
      for (int k=0; k < coords_.size(); k++)
      {

        mesh_->interpolate(gpoints[k],coords_[k],idx);
        mesh_->derivate(coords_[k],idx,Jv_);

        if (dim_ == 3)
        {
          J_[0] = Jv_[0].x();
          J_[1] = Jv_[0].y();
          J_[2] = Jv_[0].z();
          J_[3] = Jv_[1].x();
          J_[4] = Jv_[1].y();
          J_[5] = Jv_[1].z();
          J_[6] = Jv_[2].x();
          J_[7] = Jv_[2].y();
          J_[8] = Jv_[2].z();    

        }
        else if (dim_ == 2)
        {
          J2_ = Cross(Jv_[0].asVector(),Jv_[1].asVector());
          J2_.normalize();
          J_[0] = Jv_[0].x();
          J_[1] = Jv_[0].y();
          J_[2] = Jv_[0].z();
          J_[3] = Jv_[1].x();
          J_[4] = Jv_[1].y();
          J_[5] = Jv_[1].z();
          J_[6] = J2_.x();
          J_[7] = J2_.y();
          J_[8] = J2_.z();    
        }
        else if (dim_ == 1)
        {
          // The same thing as for the surface but then for a curve.
          // Again this matrix should have a positive determinant as well. It actually
          // has an internal degree of freedom, which is not being used.
          Jv_[0].asVector().find_orthogonal(J1_,J2_);
          J_[0] = Jv_[0].x();
          J_[1] = Jv_[0].y();
          J_[2] = Jv_[0].z();
          J_[3] = J1_.x();
          J_[4] = J1_.y();
          J_[5] = J1_.z();
          J_[6] = J2_.x();
          J_[7] = J2_.y();
          J_[8] = J2_.z();          
        }
        gweights[k] = weights_[k]*InverseMatrix3x3(J_, Ji_)*vol_;
      }
    
    
    }

  private:
    FIELD*                                 field_;
    typename FIELD::mesh_type*             mesh_;
    typename FIELD::mesh_type::basis_type  basis_;
    GAUSSIAN gauss_;

    std::vector<std::vector<double> > coords_;
    std::vector<double> weights_;
    double vol_;
    int    dim_;  
    
    std::vector<Point> Jv_;
    double J_[9], Ji_[9];
    Vector J1_, J2_;
};


template <class FIELD, int SIZE >
class RegularIntegration 
{
  public:

    RegularIntegration(FIELD* field)
    {
      field_ = field;
      if (field_)
      {
        mesh_  = field->get_typed_mesh().get_rep();
        basis_ = mesh_->get_basis();
        
        dim_ = basis_.domain_dimension();
        if (dim_ == 1)
        {
          coords_.resize(SIZE);
          weights_.resize(SIZE);
          
          for (int p=0; p<SIZE; p++)
          {
            coords_[p].push_back((0.5+p)/SIZE);
            weights_[p] = 1/SIZE;
          }          
        }
        
        if (dim_ == 2)
        {
          coords_.resize(SIZE*SIZE);
          weights_.resize(SIZE*SIZE);
          
          for (int p=0; p<SIZE; p++)
          {
            for (int q=0; q<SIZE; q++)
            {
              coords_[p+q*SIZE].push_back((0.5+p)/SIZE);
              coords_[p+q*SIZE].push_back((0.5+q)/SIZE);
              weights_[p+q*SIZE] = 1/(SIZE*SIZE);
            }
          }         
        }

        if (dim_ == 3)
        {
          coords_.resize(SIZE*SIZE*SIZE);
          weights_.resize(SIZE*SIZE*SIZE);
          
          for (int p=0; p<SIZE; p++)
          {
            for (int q=0; q<SIZE; q++)
            {
              for (int r=0; r<SIZE; r++)
              {
                coords_[p+q*SIZE+r*SIZE*SIZE].push_back((0.5+p)/SIZE);
                coords_[p+q*SIZE+r*SIZE*SIZE].push_back((0.5+q)/SIZE);
                coords_[p+q*SIZE+r*SIZE*SIZE].push_back((0.5+r)/SIZE);
                weights_[p+q*SIZE+r*SIZE*SIZE] = 1/(SIZE*SIZE*SIZE);
              }
            }
          }         
        }
      }  
    }

    void get_nodes_and_weights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<double>& gweights)
    {    
      gpoints.resize(weights_.size());
      gweights.resize(weights_.size());
      
      for (int k=0; k < weights_.size(); k++)
      {
        mesh_->interpolate(gpoints[k],coords_[k],idx);
        gweights[k] = weights_[k];
      }
    }

    void get_nodes_and_iweights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<double>& gweights)
    {    
      gpoints.resize(weights_.size());
      gweights.resize(weights_.size());
      
      for (int k=0; k < weights_.size(); k++)
      {
        mesh_->interpolate(gpoints[k],coords_[k],idx);

        if (dim_ == 3)
        {
          J_[0] = Jv_[0].x();
          J_[1] = Jv_[0].y();
          J_[2] = Jv_[0].z();
          J_[3] = Jv_[1].x();
          J_[4] = Jv_[1].y();
          J_[5] = Jv_[1].z();
          J_[6] = Jv_[2].x();
          J_[7] = Jv_[2].y();
          J_[8] = Jv_[2].z();    

        }
        else if (dim_ == 2)
        {
          J2_ = Cross(Jv_[0].asVector(),Jv_[1].asVector());
          J2_.normalize();
          J_[0] = Jv_[0].x();
          J_[1] = Jv_[0].y();
          J_[2] = Jv_[0].z();
          J_[3] = Jv_[1].x();
          J_[4] = Jv_[1].y();
          J_[5] = Jv_[1].z();
          J_[6] = J2_.x();
          J_[7] = J2_.y();
          J_[8] = J2_.z();    
        }
        else if (dim_ == 1)
        {
          // The same thing as for the surface but then for a curve.
          // Again this matrix should have a positive determinant as well. It actually
          // has an internal degree of freedom, which is not being used.
          Jv_[0].asVector().find_orthogonal(J1_,J2_);
          J_[0] = Jv_[0].x();
          J_[1] = Jv_[0].y();
          J_[2] = Jv_[0].z();
          J_[3] = J1_.x();
          J_[4] = J1_.y();
          J_[5] = J1_.z();
          J_[6] = J2_.x();
          J_[7] = J2_.y();
          J_[8] = J2_.z();          
        }
        gweights[k] = weights_[k]*InverseMatrix3x3(J_, Ji_);
      }
    }


  private:
    FIELD*                                 field_;
    typename FIELD::mesh_type*             mesh_;
    typename FIELD::mesh_type::basis_type  basis_;

    std::vector<std::vector<double> > coords_;
    std::vector<double> weights_;

    double vol_;
    int    dim_;  
    
    std::vector<Point> Jv_;
    double J_[9], Ji_[9];
    Vector J1_, J2_;
};

}

#endif

