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
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/include/Mesquite.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/MeshImpl.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/MeshWriter.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/MsqError.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/InstructionQueue.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/SmartLaplacianSmoother.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/TerminationCriterion.hpp"
#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/TopologyInfo.hpp"
//#include "/home/sci/jfsheph/Mesquite-and-Verdict/mesquite-1.1.3/includeLinks/TSTTUtil.hpp"

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
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  return field;

//  CubitStatus MesquiteSmoother::smart_laplacian_smooth_entities( DLIList<MRefEntity*> &entities)
//   {
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
//       PRINT_INFO("\n   Mesquite:  %s\n",err.error_message());
//       return CUBIT_FAILURE;  
    return field;
  }
    
//     for (size_t i = 0; i < ((size_t) entities.size()); i++)
//     {
      
//       if( CubitMessage::instance()->Interrupt() ) 
//       {
//         PRINT_WARNING( "----- Smoothing interrupted ----\n\n" );
//         return CUBIT_FAILURE;
//       }
      
      
//       MRefEntity* cur_ent= entities.get_and_step();
//       if(cur_ent==NULL){
//         PRINT_ERROR("Mesquite Smoother recieved null pointer to entity.\n");
//         return CUBIT_FAILURE;
//       }
//       else if(!cur_ent->is_meshed()){
//         PRINT_WARNING( "Smoother called for %s (%s %d) which is not meshed\n",
//                        cur_ent->entity_name().c_str(),
//                        cur_ent->class_name(), cur_ent->id() );
//       }
//       else{
//           // Create a Mesh object

//NOTE TO JS: 
  Mesquite::MeshImpl *mesh = new Mesquite::MeshImpl;
//  MsqPrintError err(cout);
  mesh->read_vtk("../../meshFiles/2D/VTK/square_quad_2.vtk", err);

//   char file_name[128];
//   /* Reads a TSTT Mesh file */
//   TSTT::Mesh_Handle mesh;
//   TSTT::MeshError tstt_err;
//   TSTT::Mesh_Create(&mesh, &tstt_err);
//   strcpy(file_name, "../../meshFiles/2D/VTK/Mesquite_geo_10242.vtk");
//   TSTT::Mesh_Load(mesh, file_name, &tstt_err);

//         MesquiteRefEntityMesh entity_mesh(cur_ent);
//           // Create a MeshDomain
//         MesquiteRefEntityDomain domain;
        
//           // run smoother
  if(err)
  {
    mod->error( "Error occured during Mesquite initizlization\n" );
//           PRINT_INFO("\n   Mesquite:  %s\n",err.error_message());
//           return CUBIT_FAILURE;
    return field;
  }
  else
  {
    queue.run_instructions(&entity_mesh, &domain, err); 
    MSQ_CHKERR(err);
    if(err)
    {
      mod-error( "Error occured during Mesquite smart laplacian smoothing.\n" );
//             PRINT_INFO("\n   Mesquite:  %s\n",err.error_message());
//             return CUBIT_FAILURE;
      return field;
    }
  }
      
  Mesquite::MeshWriter::write_vtk( mesh, "smoothed_mesh", err);
  if (err) return field;
   
//         CubitObserver::notify_static_observers(
//             dynamic_cast<CubitObservable*>(cur_ent),
//             MESH_MODIFIED);
//       }
//     }
//     return CUBIT_SUCCESS;
//   }
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
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  return field;
}


} // end namespace SCIRun

#endif // MeshSmoother_h
