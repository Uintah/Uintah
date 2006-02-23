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


//    File   : MeshSmoother.h
//    Author : Jason Shepherd
//    Date   : January 2006

#if !defined(MeshSmoother_h)
#define MeshSmoother_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <sci_hash_map.h>
#include <algorithm>
#include <set>
#include <Mesquite.hpp>
#include <MeshImpl.hpp>
#include <MeshWriter.hpp>
#include <MsqError.hpp>
#include <InstructionQueue.hpp>
#include <SmartLaplacianSmoother.hpp>
#include <TerminationCriterion.hpp>
#include <TopologyInfo.hpp>
#include <Dataflow/Modules/Fields/MesquiteMesh.h>
#include <Dataflow/Modules/Fields/MesquiteDomain.h>

namespace SCIRun {

using std::copy;

class GuiInterface;

class MeshSmootherAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};


template <class FIELD>
class MeshSmootherAlgoTet : public MeshSmootherAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshSmootherAlgoTet<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{
//need to make a copy of the field so that this one is not damaged...
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  double cull_eps=1e-4;
  Mesquite::MsqError err;
  Mesquite::SmartLaplacianSmoother sl_smoother(NULL,err);
    //Set stopping criterion
  Mesquite::TerminationCriterion term_inner, term_outer;
  term_inner.set_culling_type( Mesquite::TerminationCriterion::VERTEX_MOVEMENT_ABSOLUTE, cull_eps, err );
  term_outer.add_criterion_type_with_int( Mesquite::TerminationCriterion::NUMBER_OF_ITERATES, 100, err );
  term_outer.add_criterion_type_with_double( Mesquite::TerminationCriterion::CPU_TIME, 600, err );
    // sets a culling method on the first QualityImprover
  sl_smoother.set_inner_termination_criterion(&term_inner);
  sl_smoother.set_outer_termination_criterion(&term_outer);
  Mesquite::InstructionQueue queue;
    
  queue.disable_automatic_quality_assessment();
  queue.disable_automatic_midnode_adjustment();
    
  queue.set_master_quality_improver(&sl_smoother, err);
    
    
  if(err)
  {
    mod->error( "Unexpected error from Mesquite code.\n" );
    return field;
  }

  MesquiteMesh<FIELD> entity_mesh( field );
    // Create a MeshDomain
  MesquiteDomain domain;
        
//           // run smoother
  if(err)
  {
    mod->error( "Error occured during Mesquite initizlization\n" );
    return field;
  }
  else
  {
    queue.run_instructions(&entity_mesh, &domain, err); 
    MSQ_CHKERR(err);
    if(err)
    {
      mod->error( "Error occured during Mesquite smart laplacian smoothing.\n" );
      return field;
    }
  }

  return field;
}

template <class FIELD>
class MeshSmootherAlgoHex : public MeshSmootherAlgo
{
public:
    //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};

template <class FIELD>
FieldHandle MeshSmootherAlgoHex<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh)
{  
//  need to make a copy of the field, so that the previous one is not damaged...
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  double cull_eps = 1e-4;
  Mesquite::MsqError err;
  Mesquite::SmartLaplacianSmoother sl_smoother(NULL,err);
    //Set stopping criterion
  Mesquite::TerminationCriterion term_inner, term_outer;
  term_inner.set_culling_type( Mesquite::TerminationCriterion::VERTEX_MOVEMENT_ABSOLUTE, cull_eps, err );
  term_outer.add_criterion_type_with_int( Mesquite::TerminationCriterion::NUMBER_OF_ITERATES, 100, err );
  term_outer.add_criterion_type_with_double( Mesquite::TerminationCriterion::CPU_TIME, 600, err );
    // sets a culling method on the first QualityImprover
  sl_smoother.set_inner_termination_criterion(&term_inner);
  sl_smoother.set_outer_termination_criterion(&term_outer);
  Mesquite::InstructionQueue queue;
    
  queue.disable_automatic_quality_assessment();
  queue.disable_automatic_midnode_adjustment();
    
  queue.set_master_quality_improver(&sl_smoother, err);  
    
  if(err)
  {
    mod->error( "Unexpected error from Mesquite code.\n" );
    return field;
  }
  cout << "Setup Mesquite mesh interface..." << endl;

  MesquiteMesh<FIELD> entity_mesh( field );
  cout << "Setup Mesquite mesh domain..." << endl;
  MesquiteDomain domain;

  cout << "Smoothing mesh..." << endl;        
  if(err)
  {
    mod->error( "Error occured during Mesquite initizlization\n" );
    return field;
  }
  else
  {
    queue.run_instructions(&entity_mesh, &domain, err); 
    MSQ_CHKERR(err);
    if(err)
    {
      mod->error( "Error occured during Mesquite smart laplacian smoothing.\n" );
      return field;
    }
  }

  return field;
}


} // end namespace SCIRun

#endif // MeshSmoother_h
