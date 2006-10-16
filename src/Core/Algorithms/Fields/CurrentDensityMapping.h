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

#ifndef CORE_ALGORITHMS_FIELDS_CURRENTDENSITYMAPPING_H
#define CORE_ALGORITHMS_FIELDS_CURRENTDENSITYMAPPING_H 1

#include <Core/Algorithms/Fields/Mapping.h>

namespace SCIRunAlgo {


using namespace SCIRun;

class CurrentDensityMappingAlgo : public DynamicAlgoBase
{
  public:
  
    virtual bool CurrentDensityMapping(ProgressReporter* pr,
                                    int numproc, FieldHandle pot_src, FieldHandle con_src,
                                    FieldHandle dst, FieldHandle& output,
                                    std::string mappingmethod,
                                    std::string integrationmethod,
                                    std::string integrationfilter,
                                    bool multiply_with_normal,
                                    bool calcnorm);      
};


template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
class CurrentDensityMappingAlgoT : public CurrentDensityMappingAlgo
{
public:
  virtual bool CurrentDensityMapping(ProgressReporter* pr,
                                    int numproc, FieldHandle pot_src, FieldHandle con_src,
                                    FieldHandle dst, FieldHandle& output,
                                    std::string mappingmethod,
                                    std::string integrationmethod,
                                    std::string integrationfilter,
                                    bool multiply_with_normal,
                                    bool calcnorm); 
                       
  class IData
  {
    public:
      ProgressReporter*         pr;
      FPOT*                     pfield;
      FCON*                     cfield;
      FOUT*                     ofield;
      typename FPOT::mesh_type* pmesh;
      typename FCON::mesh_type* cmesh;
      typename FOUT::mesh_type* omesh;
      int                       numproc;
      std::string               integrationfilter;
      bool                      multiply_with_normal;
      bool                      retval;
  };
  
  void parallel(int procnum,IData* inputdata); 
};

template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
class CurrentDensityMappingNormalAlgoT : public CurrentDensityMappingAlgo
{
public:
  virtual bool CurrentDensityMapping(ProgressReporter* pr,
                                    int numproc, FieldHandle pot_src, FieldHandle con_src,
                                    FieldHandle dst, FieldHandle& output,
                                    std::string mappingmethod,
                                    std::string integrationmethod,
                                    std::string integrationfilter,
                                    bool multiply_with_normal,
                                    bool calcnorm); 
                       
  class IData
  {
    public:
      ProgressReporter*         pr;
      FPOT*                     pfield;
      FCON*                     cfield;
      FOUT*                     ofield;
      typename FPOT::mesh_type* pmesh;
      typename FCON::mesh_type* cmesh;
      typename FOUT::mesh_type* omesh;
      int                       numproc;
      std::string               integrationfilter;
      bool                      multiply_with_normal;
      bool                      retval;
  };
  
  void parallel(int procnum,IData* inputdata); 
};

template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
class CurrentDensityMappingNormAlgoT : public CurrentDensityMappingAlgo
{
public:
  virtual bool CurrentDensityMapping(ProgressReporter* pr,
                                    int numproc, FieldHandle pot_src, FieldHandle con_src,
                                    FieldHandle dst, FieldHandle& output,
                                    std::string mappingmethod,
                                    std::string integrationmethod,
                                    std::string integrationfilter,
                                    bool multiply_with_normal,
                                    bool calcnorm); 
                       
  class IData
  {
    public:
      ProgressReporter*         pr;
      FPOT*                     pfield;
      FCON*                     cfield;
      FOUT*                     ofield;
      typename FPOT::mesh_type* pmesh;
      typename FCON::mesh_type* cmesh;
      typename FOUT::mesh_type* omesh;
      int                       numproc;
      std::string               integrationfilter;
      bool                      multiply_with_normal;
      bool                      retval;
  };
  
  void parallel(int procnum,IData* inputdata); 
};



template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
bool CurrentDensityMappingAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::CurrentDensityMapping(ProgressReporter *pr,
                       int numproc, FieldHandle pot, FieldHandle con,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool multiply_with_normal,
                       bool calcnorm) 
{
  FPOT* pfield = dynamic_cast<FPOT*>(pot.get_rep());
  if (pfield == 0)
  {
    pr->error("CurrentDensityMapping: No input potential field was given");
    return (false);
  }

  typename FPOT::mesh_type* pmesh = dynamic_cast<typename FPOT::mesh_type*>(pot->mesh().get_rep());
  if (pmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input potential field");
    return (false);  
  }

  FCON* cfield = dynamic_cast<FCON*>(con.get_rep());
  if (cfield == 0)
  {
    pr->error("CurrentDensityMapping: No input conductivity field was given");
    return (false);
  }

  typename FCON::mesh_type* cmesh = dynamic_cast<typename FCON::mesh_type*>(con->mesh().get_rep());
  if (cmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input conductivity field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("CurrentDensityMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("CurrentDensityMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(dst.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.pfield = pfield;
  IData.cfield = cfield;
  IData.ofield = ofield;
  IData.pmesh  = pmesh;
  IData.cmesh  = cmesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  IData.integrationfilter = integrationfilter;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&CurrentDensityMappingAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::parallel,np,&IData);
    
  return (IData.retval);
}



template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
bool CurrentDensityMappingNormAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::CurrentDensityMapping(ProgressReporter *pr,
                       int numproc, FieldHandle pot, FieldHandle con,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool multiply_with_normal,
                       bool calcnorm) 
{
  FPOT* pfield = dynamic_cast<FPOT*>(pot.get_rep());
  if (pfield == 0)
  {
    pr->error("CurrentDensityMapping: No input potential field was given");
    return (false);
  }

  typename FPOT::mesh_type* pmesh = dynamic_cast<typename FPOT::mesh_type*>(pot->mesh().get_rep());
  if (pmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input potential field");
    return (false);  
  }

  FCON* cfield = dynamic_cast<FCON*>(con.get_rep());
  if (cfield == 0)
  {
    pr->error("CurrentDensityMapping: No input conductivity field was given");
    return (false);
  }

  typename FCON::mesh_type* cmesh = dynamic_cast<typename FCON::mesh_type*>(con->mesh().get_rep());
  if (cmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input conductivity field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("CurrentDensityMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("CurrentDensityMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(dst.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.pfield = pfield;
  IData.cfield = cfield;
  IData.ofield = ofield;
  IData.pmesh  = pmesh;
  IData.cmesh  = cmesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  IData.integrationfilter = integrationfilter;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&CurrentDensityMappingNormAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::parallel,np,&IData);
    
  return (IData.retval);
}



template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
void CurrentDensityMappingAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Elem::iterator it, eit;
  typename FOUT::mesh_type::Elem::size_type s;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val, val2;
  FOUT* ofield = idata->ofield;
  
  int numproc = idata->numproc;
  ProgressReporter *pr = idata->pr;
  
  omesh->begin(it);
  omesh->end(eit);
  omesh->size(s);
  
  int cnt = 0;
  
  InterpolatedGradient<FPOT> pmapping(idata->pfield);
  InterpolatedData<FCON> cmapping(idata->cfield);

  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::string filter = idata->integrationfilter;
  double con;
  typename FOUT::value_type grad;
  Vector g;
  
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
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          valarray[p] = -(con*g);
        }
        else
        {
          valarray[p] = 0;
        }
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
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
        if (pmapping.get_gradient(points[0],grad)&&cmapping.get_data(points[0],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val = -(con*g); 
        }
        else
        {
          val = 0;
        }
        for (size_t p = 1; p < points.size(); p++)
        {
          if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
          {
            g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
            tval = -(con*g); 
          }
          else
          {
            tval = 0;
          }
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
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
        if (pmapping.get_gradient(points[0],grad)&&cmapping.get_data(points[0],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val = -(con*g); 
        }
        else
        {
          val = 0;
        }
        for (size_t p = 1; p < points.size(); p++)
        {
          if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
          {
            g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
            tval = -(con*g); 
          }
          else
          {
            tval = 0;
          }
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
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
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          valarray[p] = -(con*g); 
        }
        else
        {
          valarray[p] = 0;
        }
      }
      sort(valarray.begin(),valarray.end());
       
      typename FOUT::value_type rval; rval = 0;
      typename FOUT::value_type val;  val = 0;
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
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_iweights(*it,points,weights);

      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val2 = -(con*g); 
        }
        else
        {
          val2 = 0;
        }
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }
  }
  else if ((filter == "weightedaverage")||(filter == "WeightedAverage"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);

      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];        
          val2 = -(con*g); 
        }
        else
        {
          val2 = 0;
        }
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];        
          val2 = -(con*g); 
        }
        else
        {
          val2 = 0;
        }
        val += val2 * (1.0/points.size());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }
  }
  else if ((filter == "sum")||(filter == "Sum"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val2 = -(con*g); 
        }
        else
        {
          val2 = 0;
        }        val += val2;
      }

      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }
  }
  else
  {
    if (procnum == 0)
    {
      idata->pr->error("CurrentDensintyMapping: Filter method is unknown");
      idata->retval = false;
    }
    return;
  }
  
  if (procnum == 0) idata->retval = true;
}








template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
void CurrentDensityMappingNormAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Elem::iterator it, eit;
  typename FOUT::mesh_type::Elem::size_type s;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val, val2;
  FOUT* ofield = idata->ofield;
  
  int numproc = idata->numproc;
  ProgressReporter *pr = idata->pr;
  
  omesh->begin(it);
  omesh->end(eit);
  omesh->size(s);
  
  int cnt = 0;
  
  InterpolatedGradient<FPOT> pmapping(idata->pfield);
  InterpolatedData<FCON> cmapping(idata->cfield);

  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::string filter = idata->integrationfilter;
  double con;
  Vector grad;
  Vector g;
  
  
  
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
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          Vector v = con*g;
          valarray[p] = static_cast<typename FOUT::value_type>(v.length());
        }
        else
        {
          valarray[p] = 0;
        }
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
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
        if (pmapping.get_gradient(points[0],grad)&&cmapping.get_data(points[0],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          Vector v = con*g;
          val = static_cast<typename FOUT::value_type>(v.length()); 
        }
        else
        {
          val = 0;
        }
        for (size_t p = 1; p < points.size(); p++)
        {
          if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
          {
            g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
            Vector v = con*g;
            tval = static_cast<typename FOUT::value_type>(v.length()); 
          }
          else
          {
            tval = 0;
          }
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
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
        if (pmapping.get_gradient(points[0],grad)&&cmapping.get_data(points[0],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          Vector v = con*g;
          val = static_cast<typename FOUT::value_type>(v.length());
        }
        else
        {
          val = 0;
        }
        for (size_t p = 1; p < points.size(); p++)
        {
          if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
          {
            g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
            Vector v = con*g;
            tval = static_cast<typename FOUT::value_type>(v.length()); 
          }
          else
          {
            tval = 0;
          }
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
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
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          Vector v = con*g;
          valarray[p] = static_cast<typename FOUT::value_type>(v.length()); 
        }
        else
        {
          valarray[p] = 0;
        }
      }
      sort(valarray.begin(),valarray.end());
       
      typename FOUT::value_type rval; rval = 0;
      typename FOUT::value_type val;  val = 0;
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
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_iweights(*it,points,weights);

      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          Vector v = con*g;
          val2 = static_cast<typename FOUT::value_type>(v.length()); 
        }
        else
        {
          val2 = 0;
        }
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else if ((filter == "weightedaverage")||(filter == "WeightedAverage"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);

      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];        
          Vector v = con*g;
          val2 = static_cast<typename FOUT::value_type>(v.length()); 
        }
        else
        {
          val2 = 0;
        }
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];        
          Vector v = con*g;
          val2 = static_cast<typename FOUT::value_type>(v.length()); 
        }
        else
        {
          val2 = 0;
        }
        val += val2 * (1.0/points.size());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }      
    }
  }
  else if ((filter == "sum")||(filter == "Sum"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_and_weights(*it,points,weights);
      
      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          Vector v = con*g;
          val2 = static_cast<typename FOUT::value_type>(v.length()); 
        }
        else
        {
          val2 = 0;
        }        
        val += val2;
      }

      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else
  {
    if (procnum == 0)
    {
      idata->pr->error("CurrentDensintyMapping: Filter method is unknown");
      idata->retval = false;
    }
    return;
  }
  
  if (procnum == 0) idata->retval = true;
}





template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
bool CurrentDensityMappingNormalAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::CurrentDensityMapping(ProgressReporter *pr,
                       int numproc, FieldHandle pot, FieldHandle con,FieldHandle dst, FieldHandle& output, 
                       std::string mappingmethod,
                       std::string integrationmethod,
                       std::string integrationfilter,
                       bool multiply_with_normal,
                       bool calcnorm) 
{
  FPOT* pfield = dynamic_cast<FPOT*>(pot.get_rep());
  if (pfield == 0)
  {
    pr->error("CurrentDensityMapping: No input potential field was given");
    return (false);
  }

  typename FPOT::mesh_type* pmesh = dynamic_cast<typename FPOT::mesh_type*>(pot->mesh().get_rep());
  if (pmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input potential field");
    return (false);  
  }

  FCON* cfield = dynamic_cast<FCON*>(con.get_rep());
  if (cfield == 0)
  {
    pr->error("CurrentDensityMapping: No input conductivity field was given");
    return (false);
  }

  typename FCON::mesh_type* cmesh = dynamic_cast<typename FCON::mesh_type*>(con->mesh().get_rep());
  if (cmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input conductivity field");
    return (false);  
  }

  FDST* dfield = dynamic_cast<FDST*>(dst.get_rep());
  if (dfield == 0)
  {
    pr->error("CurrentDensityMapping: No input destination field was given");
    return (false);
  }

  typename FDST::mesh_type* dmesh = dynamic_cast<typename FDST::mesh_type*>(dst->mesh().get_rep());
  if (dmesh == 0)
  {
    pr->error("CurrentDensityMapping: No mesh is associated with input destination field");
    return (false);  
  }

  output = dynamic_cast<Field *>(scinew FOUT(dmesh));
  if (output.get_rep() == 0)
  {
    pr->error("CurrentDensityMapping: Could no allocate output field");
    return (false);
  }
  
  FOUT* ofield = dynamic_cast<FOUT*>(output.get_rep());
  ofield->resize_fdata();
  
  output->copy_properties(dst.get_rep());

  // Now do parallel algorithm

  IData IData;
  IData.pfield = pfield;
  IData.cfield = cfield;
  IData.ofield = ofield;
  IData.pmesh  = pmesh;
  IData.cmesh  = cmesh;
  IData.omesh  = dmesh;
  IData.pr     = pr;
  IData.integrationfilter = integrationfilter;
  
  // Determine the number of processors to use:
  int np = Thread::numProcessors(); if (np > 5) np = 5;  
  if (numproc > 0) { np = numproc; }
  IData.numproc = np;
   
  Thread::parallel(this,&CurrentDensityMappingNormalAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::parallel,np,&IData);
    
  return (IData.retval);
}



template <class INTEGRATOR, class FPOT, class FCON, class FDST, class FOUT>
void CurrentDensityMappingNormalAlgoT<INTEGRATOR,FPOT,FCON,FDST,FOUT>::parallel(int procnum,IData* idata)
{
  typename FOUT::mesh_type::Elem::iterator it, eit;
  typename FOUT::mesh_type::Elem::size_type s;
  typename FOUT::mesh_type* omesh = idata->omesh;
  typename FOUT::value_type val, val2;
  FOUT* ofield = idata->ofield;
  
  int numproc = idata->numproc;
  ProgressReporter *pr = idata->pr;
  
  omesh->begin(it);
  omesh->end(eit);
  omesh->size(s);
  
  int cnt = 0;
  
  InterpolatedGradient<FPOT> pmapping(idata->pfield);
  InterpolatedData<FCON> cmapping(idata->cfield);

  INTEGRATOR integrator(idata->ofield);
  
  for (int p =0; p < procnum; p++) if (it != eit) ++it;
  std::vector<Point> points;
  std::vector<double> weights;
  std::vector<Vector> normals;
  std::string filter = idata->integrationfilter;
  double con;
  Vector grad;
  Vector g;
  
  // Determine the filter and loop over nodes
  if ((filter == "median")||(filter == "Median"))
  {
    // median filter over integration nodes
    std::vector<typename FOUT::value_type> valarray;
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);
      valarray.resize(points.size());

      for (size_t p = 0; p < points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          valarray[p] = -Dot(normals[p],con*g);         
        }
        else
        {
          valarray[p] = 0; 
        }
      }
      sort(valarray.begin(),valarray.end());
      int idx = static_cast<int>((valarray.size()/2));
      ofield->set_value(valarray[idx],*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it; 
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }       
    }
  }
  else if ((filter == "minimum")||(filter == "Minimum"))
  {
    // minimum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);
      typename FOUT::value_type val = 0;
      typename FOUT::value_type tval = 0;

      if (points.size() > 0)
      {
        if (pmapping.get_gradient(points[0],grad)&&cmapping.get_data(points[0],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val = -Dot(normals[0],con*g);         
        }
        else
        {
          val = 0;
        }
        for (size_t p = 1; p < points.size(); p++)
        {
          if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
          {
            g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
            tval = -Dot(normals[p],con*g);         
          }
          else
          {
            tval = 0;
          }
          if (tval < val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }  
  }
  else if ((filter == "maximum")||(filter == "Maximum"))
  {
    // maximum filter over integration nodes
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);
      typename FOUT::value_type val = 0;
      typename FOUT::value_type tval = 0;

      if (points.size() > 0)
      {
        if (pmapping.get_gradient(points[0],grad)&&cmapping.get_data(points[0],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val = -Dot(normals[0],con*g);         
        }
        else
        {
          val = 0;
        }

        for (size_t p = 1; p < points.size(); p++)
        {
          if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
          {
            g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
            tval = -Dot(normals[p],con*g);         
          }
          else
          {
            tval = 0;
          }
          if (tval > val) val = tval;
        }
      }
      ofield->set_value(val,*it);
      
      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else if ((filter == "mostcommon")||(filter == "Mostcommon")||(filter == "MostCommon"))
  {
    // Filter designed for segmentations where one wants the most common element to be the
    // sampled element
    std::vector<typename FOUT::value_type> valarray;
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);
      valarray.resize(points.size());
      for (size_t p = 0; p < points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          valarray[p] = -Dot(normals[p],con*g);         
        }
        else
        {
          valarray[p] = 0;
        }
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
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }  
  }
  else if ((filter == "integrate")||(filter == "Integrate"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_normals_and_iweights(*it,points,normals,weights);

      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val2 = -Dot(normals[p],con*g);         
        }
        else
        {
          val2 = 0;
        }
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else if ((filter == "weightedaverage")||(filter == "WeightedAverage"))
  {
    // Real integration of underlying value
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);

      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val2 = -Dot(normals[p],con*g);         
        }
        else
        {
          val2 = 0;
        }
        val += val2*weights[p];
      }
      
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else if ((filter == "average")||(filter == "Average"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);
      
      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
          g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
          val2 = -Dot(normals[p],con*g);         
        }
        else
        {
          val2 = 0;
        }
        val += val2 * (1.0/points.size());
      }
      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else if ((filter == "sum")||(filter == "Sum"))
  {
    // Average, like integrate but ignore weights
    while (it != eit)
    {
      integrator.get_nodes_normals_and_weights(*it,points,normals,weights);
      
      val = 0;
      for (int p=0; p<points.size(); p++)
      {
        if (pmapping.get_gradient(points[p],grad)&&cmapping.get_data(points[p],con))
        {
           g[0] = grad[0]; g[1] = grad[1]; g[2] = grad[2];
           val2 = -Dot(normals[p],con*g); 
        }
        else
        {
          val2 = 0;
        }
        val += val2;
      }

      ofield->set_value(val,*it);

      for (int p =0; p < numproc; p++) if (it != eit) ++it;  
      if (procnum == 0) { cnt++; if (cnt == 100) { pr->update_progress(static_cast<int>(*it),static_cast<int>(s)); cnt = 1; } }
    }
  }
  else
  {
    if (procnum == 0)
    {
      idata->pr->error("CurrentDensintyMapping: Filter method is unknown");
      idata->retval = false;
    }
    return;
  }
  
  if (procnum == 0) idata->retval = true;
}







template <class GAUSSIAN, class FIELD >
class NormalGaussianIntegration 
{
  public:

    inline NormalGaussianIntegration(FIELD* field)
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
        
        dim_ = basis_.domain_dimension();
        if (dim_ == 3)
        {
          vol_ = basis_.volume();
        }
        else if (dim_ == 2)
        {
          vol_ = basis_.area(0);
        }
        else if (dim_ == 0)
        {
          vol_ = basis_.length(0);
        }
        else
        {
          vol_ = 0.0;
        }
      }
    }

    inline void get_nodes_normals_and_weights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<Vector>& gnormals, std::vector<double>& gweights)
    {    
      gpoints.resize(gauss_.GaussianNum);
      gweights.resize(gauss_.GaussianNum);
      gnormals.resize(gauss_.GaussianNum);
      
      for (int k=0; k < coords_.size(); k++)
      {
        mesh_->interpolate(gpoints[k],coords_[k],idx);
        mesh_->get_normal(gnormals[k],coords_[k],idx,0);
        gweights[k] = weights_[k];
      }
    }
        
    inline void get_nodes_normals_and_iweights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<Vector>& gnormals, std::vector<double>& gweights)
    {    
      
      gpoints.resize(gauss_.GaussianNum);
      gweights.resize(gauss_.GaussianNum);
      gnormals.resize(gauss_.GaussianNum);
      
      for (int k=0; k < coords_.size(); k++)
      {

        mesh_->interpolate(gpoints[k],coords_[k],idx);
        mesh_->get_normal(gnormals[k],coords_[k],idx,0);
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
class NormalRegularIntegration 
{
  public:

    inline NormalRegularIntegration(FIELD* field)
    {
      field_ = field;
      if (field_)
      {
        mesh_  = field->get_typed_mesh().get_rep();
        basis_ = mesh_->get_basis();
        
        vol_ = 0.0;
        
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
          vol_ = 1.0;
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
          vol_ = 1.0;
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
          vol_ = 1.0;         
        }
      }  
    }

    inline void get_nodes_normals_and_weights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<Vector>& gnormals, std::vector<double>& gweights)
    {    
      gpoints.resize(weights_.size());
      gweights.resize(weights_.size());
      gnormals.resize(weights_.size());
      
      for (int k=0; k < weights_.size(); k++)
      {
        mesh_->interpolate(gpoints[k],coords_[k],idx);
        mesh_->get_normal(gnormals[k],coords_[k],idx,0);
        gweights[k] = weights_[k];
      }
    }

    inline void get_nodes_normals_and_iweights(typename FIELD::mesh_type::Elem::index_type idx, std::vector<Point>& gpoints, std::vector<Vector>& gnormals, std::vector<double>& gweights)
    {    
      gpoints.resize(weights_.size());
      gweights.resize(weights_.size());
      gnormals.resize(weights_.size());
      
      for (int k=0; k < weights_.size(); k++)
      {
        mesh_->interpolate(gpoints[k],coords_[k],idx);
        mesh_->get_normal(gnormals[k],coords_[k],idx,0);
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



} // end namespace

#endif

