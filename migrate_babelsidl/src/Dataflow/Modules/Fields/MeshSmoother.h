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
#include <Dataflow/Modules/Fields/QuadToTri.h>
#include <sci_hash_map.h>
#include <algorithm>
#include <set>
#include <Mesquite.hpp>
#include <MeshImpl.hpp>
#include <MeshWriter.hpp>
#include <MsqError.hpp>
#include <InstructionQueue.hpp>
#include <LaplacianSmoother.hpp>
#include <SmartLaplacianSmoother.hpp>
#include <TerminationCriterion.hpp>
#include <TopologyInfo.hpp>
#include <UntangleBetaQualityMetric.hpp>
#include <TerminationCriterion.hpp>
#include <IdealWeightInverseMeanRatio.hpp>
#include <LPtoPTemplate.hpp>
#include <FeasibleNewton.hpp>
#include <ConjugateGradient.hpp>
#include <Dataflow/Modules/Fields/MesquiteMesh.h>
#include <Dataflow/Modules/Fields/MesquiteDomain.h>
#include <ctime>

namespace SCIRun {

using std::copy;

class GuiInterface;

class MeshSmootherAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute( ProgressReporter *reporter, FieldHandle fieldh, 
                               bool boundary, string scheme ) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);
};


template <class FIELD>
class MeshSmootherAlgoShared : public MeshSmootherAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute( ProgressReporter *reporter, FieldHandle fieldh,
                               bool boundary, string scheme );

protected:
  virtual FieldHandle compute_boundary( ProgressReporter *reporter,
                                        FieldHandle fieldh,
                                        string scheme ) = 0;

  virtual bool compute_domain_surface( ProgressReporter *reporter,
                                       FieldHandle fieldh,
                                       FieldHandle &ofieldh);

  FieldHandle laplacian_smoother( ProgressReporter *reporter,
                                  FieldHandle fieldh );  
  FieldHandle smart_laplacian_smoother( ProgressReporter *reporter,
                                        FieldHandle fieldh );
  FieldHandle shape_improvement_wrapper( ProgressReporter *reporter,
                                         FieldHandle fieldh );
};


template <class FIELD>
FieldHandle
MeshSmootherAlgoShared<FIELD>::execute( ProgressReporter *reporter,
                                        FieldHandle fieldh,
                                        bool boundary,
                                        string scheme )
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep() );
  typename FIELD::mesh_type::Elem::size_type num_elems;
  field->get_typed_mesh()->size( num_elems );
  if( num_elems == 0 )
  {
    reporter->error( "No elements to smooth." );
    return fieldh;
  }
  
  if ( boundary )
  {
    fieldh = compute_boundary( reporter, fieldh, scheme );
  }
  
  if( scheme == "None" )
  {
    return fieldh;
  }
  else if( scheme == "Laplacian" )
  { 
    return laplacian_smoother( reporter, fieldh );
  }
  else if( scheme == "SmartLaplacian" )
  { 
    return smart_laplacian_smoother( reporter, fieldh );
  }
  else if( scheme == "ShapeImprovement" )
  {
    return shape_improvement_wrapper( reporter, fieldh );
  }
  else
  {
    reporter->error( "Unknown Smoothing Scheme." );
    return fieldh;
  }
}


template <class FIELD>
bool
MeshSmootherAlgoShared<FIELD>::compute_domain_surface( ProgressReporter *mod,
                                                       FieldHandle fieldh,
                                                       FieldHandle &ofieldh)
{
  ofieldh = 0;
  return true;
}


template <class FIELD>
FieldHandle
MeshSmootherAlgoShared<FIELD>::laplacian_smoother( ProgressReporter *mod,
                                                   FieldHandle fieldh )
{
  // Need to make a copy of the field, so that the previous one is not damaged.
  FIELD *field = dynamic_cast<FIELD*>( fieldh.get_rep() );
  FIELD *ofield = scinew FIELD( field->get_typed_mesh() );
  ofield->copy_properties( fieldh.get_rep() );
  ofield->mesh_detach();
  
  Mesquite::MsqError err;
  Mesquite::IdealWeightInverseMeanRatio inverse_mean_ratio( err );

  Mesquite::LaplacianSmoother* lapl1 = new Mesquite::LaplacianSmoother( err );

//  Mesquite::QualityAssessor mQA( inverse_mean_ratio, Mesquite::QualityAssessor::MAXIMUM, err );
//  Mesquite::QualityAssessor *mQA = new MeshQualityAssessor( inverse_mean_ratio, 4, err );
  
  // Set stopping criterion.
  Mesquite::TerminationCriterion *m_term = new Mesquite::TerminationCriterion();
  m_term->add_criterion_type_with_int( Mesquite::TerminationCriterion::NUMBER_OF_ITERATES, 10, err );
  lapl1->set_outer_termination_criterion( m_term );
  
  Mesquite::InstructionQueue iqueue;
//  iqueue.add_quality_assessor( mQA, err ); 
//  MSQ_CHKERR(err);
  iqueue.set_master_quality_improver( lapl1, err );
//  MSQ_CHKERR(err);
//  iqueue.add_quality_assessor( mQA, err ); 
//  MSQ_CHKERR(err);  

  if ( err )
  {
    cout << err << endl;
    mod->error( "Unexpected error from Mesquite code." );
    return field;
  }

  MesquiteMesh<FIELD> entity_mesh( ofield, mod );  

  FieldHandle domain_field_h;
  if (!compute_domain_surface( mod, fieldh, domain_field_h ))
  {
    return false;
  }
  TriSurfMesh<TriLinearLgn<Point> > *domain_mesh = 0;
  if (domain_field_h.get_rep())
  {
    domain_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(domain_field_h->mesh().get_rep());
  }

  if( domain_mesh )
  {
    domain_mesh->synchronize(Mesh::EDGES_E | Mesh::NORMALS_E | Mesh::LOCATE_E);
    MesquiteDomain domain( domain_mesh );
    iqueue.run_instructions( &entity_mesh, &domain, err ); 
  }
  else
  {
    iqueue.run_instructions( &entity_mesh, err ); 
  }

  MSQ_CHKERR(err);
  if (err)
  {
    mod->error( "Error occurred during Mesquite Laplacian smoothing." );
    cout << err << endl;
    return field;
  }

  return ofield;
}


template <class FIELD>
FieldHandle
MeshSmootherAlgoShared<FIELD>::smart_laplacian_smoother( ProgressReporter *mod,
                                                         FieldHandle fieldh )
{
  // Need to make a copy of the field, so that the previous one is not damaged.
  FIELD *field = dynamic_cast<FIELD*>( fieldh.get_rep() );
  FIELD *ofield = scinew FIELD( field->get_typed_mesh() );
  ofield->copy_properties( fieldh.get_rep() );
  ofield->mesh_detach();
  
  double cull_eps = 1e-4;
  Mesquite::MsqError err;
  Mesquite::SmartLaplacianSmoother sl_smoother(NULL,err);
  // Set stopping criterion.
  Mesquite::TerminationCriterion term_inner, term_outer;
  term_inner.set_culling_type( Mesquite::TerminationCriterion::VERTEX_MOVEMENT_ABSOLUTE, cull_eps, err );
  term_outer.add_criterion_type_with_int( Mesquite::TerminationCriterion::NUMBER_OF_ITERATES, 100, err );
  term_outer.add_criterion_type_with_double( Mesquite::TerminationCriterion::CPU_TIME, 600, err );
  // Sets a culling method on the first QualityImprover.
  sl_smoother.set_inner_termination_criterion(&term_inner);
  sl_smoother.set_outer_termination_criterion(&term_outer);
  Mesquite::InstructionQueue iqueue;
    
  iqueue.disable_automatic_quality_assessment();
  iqueue.disable_automatic_midnode_adjustment();
    
  iqueue.set_master_quality_improver( &sl_smoother, err );  
    
  if ( err )
  {
    cout << err << endl;
    mod->error( "Unexpected error from Mesquite code." );
    return field;
  }

  MesquiteMesh<FIELD> entity_mesh( ofield, mod );  

  FieldHandle domain_field_h;
  if (!compute_domain_surface( mod, fieldh, domain_field_h ))
  {
    return false;
  }
  TriSurfMesh<TriLinearLgn<Point> > *domain_mesh = 0;
  if (domain_field_h.get_rep())
  {
    domain_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(domain_field_h->mesh().get_rep());
  }

  // TODO: This looks unreachable.  err is unused since we last checked it.
  if (err)
  {
    cout << err << endl;
    mod->error( "Error occurred during Mesquite initialization" );
    return field;
  }

  if (domain_mesh)
  {
    domain_mesh->synchronize(Mesh::EDGES_E | Mesh::NORMALS_E | Mesh::LOCATE_E);
    MesquiteDomain domain( domain_mesh );
    iqueue.run_instructions( &entity_mesh, &domain, err ); 
  }
  else
  {
    iqueue.run_instructions( &entity_mesh, err ); 
  }

  MSQ_CHKERR(err);
  if (err)
  {
    mod->error( "Error occurred during Mesquite smart laplacian smoothing." );
    cout << err << endl;
    return field;
  }

  return ofield;
}


template <class FIELD>
FieldHandle
MeshSmootherAlgoShared<FIELD>::shape_improvement_wrapper(ProgressReporter *mod,
                                                         FieldHandle fieldh )
{
  // Need to make a copy of the field so that this one is not damaged.
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  FIELD *ofield = scinew FIELD( field->get_typed_mesh() );
  ofield->copy_properties( fieldh.get_rep() );
  ofield->mesh_detach();

  // Arbitrary defined variables.
  double untangle_beta_val=1e-8;
  double unt_successive_eps=1e-10;
  double abs_successive_eps=1.e-3;
  double grad_norm=1.e-6;
  double max_cpu_time=300; // 5 minutes
  clock_t start = clock();
  
  Mesquite::MsqError err;
  Mesquite::InstructionQueue queue_untangle;
  Mesquite::InstructionQueue queue_shape;

  queue_untangle.disable_automatic_quality_assessment();
  queue_untangle.disable_automatic_midnode_adjustment();
  queue_shape.disable_automatic_quality_assessment();
  queue_shape.disable_automatic_midnode_adjustment();
  
  // Set up untangle metric and function.
  Mesquite::UntangleBetaQualityMetric untangle_metric(untangle_beta_val);
  Mesquite::LPtoPTemplate untangle_function(&untangle_metric, 2, err);
  untangle_function.set_gradient_type(
                                      Mesquite::ObjectiveFunction::ANALYTICAL_GRADIENT);
  // Use cg for the untangle function.
  Mesquite::ConjugateGradient untangle_opt(&untangle_function, err);

  untangle_opt.set_patch_type(Mesquite::PatchData::GLOBAL_PATCH, err,1 ,1);
  Mesquite::TerminationCriterion untangle_inner, untangle_outer;
  untangle_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::QUALITY_IMPROVEMENT_ABSOLUTE,0.0,err);
  untangle_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::CPU_TIME,max_cpu_time,err);
  untangle_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::SUCCESSIVE_IMPROVEMENTS_ABSOLUTE,unt_successive_eps,err);
  untangle_outer.add_criterion_type_with_int(Mesquite::TerminationCriterion::NUMBER_OF_ITERATES,1,err);
  untangle_opt.set_inner_termination_criterion(&untangle_inner);
  untangle_opt.set_outer_termination_criterion(&untangle_outer);

  // Set up shape metric and function.
  Mesquite::IdealWeightInverseMeanRatio mean_ratio(err);
  
  mean_ratio.set_averaging_method(Mesquite::QualityMetric::SUM, err);
  Mesquite::LPtoPTemplate obj_func(&mean_ratio, 1, err);
  obj_func.set_gradient_type( Mesquite::ObjectiveFunction::ANALYTICAL_GRADIENT);
  obj_func.set_dividing_by_n(true);

  // Use feasible newton.
  Mesquite::FeasibleNewton shape_opt( &obj_func );
  shape_opt.set_patch_type(Mesquite::PatchData::GLOBAL_PATCH, err);
  Mesquite::TerminationCriterion shape_inner, shape_outer;
  shape_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::GRADIENT_L2_NORM_ABSOLUTE,grad_norm,err);
  shape_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::SUCCESSIVE_IMPROVEMENTS_ABSOLUTE,abs_successive_eps,err);
  shape_outer.add_criterion_type_with_int(Mesquite::TerminationCriterion::NUMBER_OF_ITERATES,1,err);
  shape_opt.set_inner_termination_criterion(&shape_inner);
  shape_opt.set_outer_termination_criterion(&shape_outer);

  // Add both to the queue.
  queue_untangle.set_master_quality_improver(&untangle_opt, err);
  queue_shape.set_master_quality_improver(&shape_opt,err);
  if(err)
  {
    cout << err << endl;
    mod->error( "Unexpected error from Mesquite code." );
    return field;  
  }
  
  untangle_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::CPU_TIME,max_cpu_time,err);

  MesquiteMesh<FIELD> entity_mesh( ofield, mod );

  FieldHandle domain_field_h;
  if (!compute_domain_surface( mod, fieldh, domain_field_h ))
  {
    return false;
  }
  TriSurfMesh<TriLinearLgn<Point> > *domain_mesh = 0;
  if (domain_field_h.get_rep())
  {
    domain_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(domain_field_h->mesh().get_rep());
  }

  // TODO: This looks almost unreachable.  err is unused since we last
  // checked it.
  if(err)
  {
    cout << err << endl;
    mod->error( "Error occurred during Mesquite initialization." );
    return field;
  } 
  clock_t finish = clock();
  double total_time = (finish - start)/(double)CLOCKS_PER_SEC;
  
  if (domain_mesh)
  {
    domain_mesh->synchronize(Mesh::EDGES_E | Mesh::NORMALS_E | Mesh::LOCATE_E);
    MesquiteDomain domain( domain_mesh );
    queue_untangle.run_instructions(&entity_mesh, &domain, err);

    MSQ_CHKERR(err);
    double time_remaining = max_cpu_time - total_time;

    if( err )
    {
      cout << err << endl;
      mod->error( "Error occurred during Mesquite untangling." );
      return field;
    }
    else if ( time_remaining <= 0.0)
    {
      mod->error( "Allotted time expired before shape optimization." );
    }
    else if (untangle_inner.get_current_function_value()>1.e-12)
    {
      mod->error( "Objective function was not zero.  Untangle unsuccessful." );
    }
    else
    {
      // Add cpu timer.
      shape_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::CPU_TIME,time_remaining,err);
      queue_shape.run_instructions(&entity_mesh, &domain, err);  
      MSQ_CHKERR(err);
      if( err )
      {
        cout << err << endl;
        mod->error( "Error occurred during smoothing operation." );
      }
    }
  }
  else
  {
    queue_untangle.run_instructions( &entity_mesh, err );
    MSQ_CHKERR(err);

    double time_remaining = max_cpu_time - total_time;
    if( err )
    {
      cout << err << endl;
      mod->error( "Error occurred during Mesquite untangling." );
      return field;
    }
    else if ( time_remaining <= 0.0)
    {
      mod->error( "Allotted time expired before shape optimization." );
    }
    else if (untangle_inner.get_current_function_value()>1.e-12)
    {
      mod->error( "Objective function was not zero.  Untangle unsuccessful." );
    }
    else
    {
      // Add cpu timer.
      shape_inner.add_criterion_type_with_double(Mesquite::TerminationCriterion::CPU_TIME,time_remaining,err);
      queue_shape.run_instructions( &entity_mesh, err );
      MSQ_CHKERR(err);
      if( err )
      {
        cout << err << endl;
        mod->error( "Error occurred during smoothing operation." );
      }
    }
  }

  return ofield;
}


template <class FIELD>
class MeshSmootherAlgoTet : public MeshSmootherAlgoShared<FIELD>
{
protected:
  //! virtual interface. 
  virtual FieldHandle compute_boundary( ProgressReporter *reporter,
                                        FieldHandle fieldh,
                                        string scheme );
};


template <class FIELD>
FieldHandle
MeshSmootherAlgoTet<FIELD>::compute_boundary( ProgressReporter *mod,
                                              FieldHandle fieldh,
                                              string scheme )
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  FIELD *ofield = scinew FIELD( field->get_typed_mesh() );
  ofield->copy_properties( fieldh.get_rep() );
  ofield->mesh_detach();
  FieldHandle ofh = ofield;
  
  const TypeDescription *mtd = ofh->mesh()->get_type_description();
  CompileInfoHandle ci_boundary = FieldBoundaryAlgo::get_compile_info( mtd );
  Handle<FieldBoundaryAlgo> boundary_algo;
  FieldHandle boundary_field_h;
  if( !DynamicCompilation::compile( ci_boundary, boundary_algo, false, mod ) )
  {
    mod->error( "Unable to obtain a list of boundary elements for smoothing the boundary." );
    return ofh;
  }

  MatrixHandle interp1(0);  
  boundary_algo->execute( mod, ofh->mesh(), boundary_field_h, interp1, 1 );
  
  const TypeDescription *std = boundary_field_h->get_type_description();
  CompileInfoHandle ci_boundary_smooth = MeshSmootherAlgo::get_compile_info( std, "Tri" );
  Handle<MeshSmootherAlgo> bound_smooth_algo;
  if (!DynamicCompilation::compile( ci_boundary_smooth, bound_smooth_algo, false, mod ) )
  {
    mod->error( "Unable to compile MeshSmoother algorithm for smoothing the boundary." );
    return ofh;
  }

  FieldHandle smoothfield = bound_smooth_algo->execute( mod, boundary_field_h, false, scheme );

  TriSurfMesh<TriLinearLgn<Point> > *smooth_boundary = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(smoothfield->mesh().get_rep());
  TriSurfMesh<TriLinearLgn<Point> >::Node::iterator bound_iter;
  smooth_boundary->begin( bound_iter );
  TriSurfMesh<TriLinearLgn<Point> >::Node::iterator bound_itere; 
  smooth_boundary->end( bound_itere );
  while( bound_iter != bound_itere )
  {
    TriSurfMesh<TriLinearLgn<Point> >::Node::index_type bi = *bound_iter;
    ++bound_iter;
      
    int size, stride, *cols;
    double *vals;
    interp1->getRowNonzerosNoCopy( bi, size, stride, cols, vals );
      
    Point p;
    smooth_boundary->get_point( p, bi );
    ofield->get_typed_mesh()->set_point( p, *cols );
  }
  return ofh;
}


template <class FIELD>
class MeshSmootherAlgoHex : public MeshSmootherAlgoShared<FIELD>
{
protected:
  //! virtual interface. 
  virtual FieldHandle compute_boundary( ProgressReporter *reporter,
                                        FieldHandle fieldh,
                                        string scheme );
};


template <class FIELD>
FieldHandle
MeshSmootherAlgoHex<FIELD>::compute_boundary( ProgressReporter *mod,
                                              FieldHandle fieldh,
                                              string scheme )
{ 
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  FIELD *ofield = scinew FIELD( field->get_typed_mesh() );
  ofield->copy_properties( fieldh.get_rep() );
  ofield->mesh_detach();
  FieldHandle ofh = ofield;

  const TypeDescription *mtd = ofh->mesh()->get_type_description();
  CompileInfoHandle ci_boundary = FieldBoundaryAlgo::get_compile_info( mtd );
  Handle<FieldBoundaryAlgo> boundary_algo;
  FieldHandle boundary_field_h;
  if( !DynamicCompilation::compile( ci_boundary, boundary_algo, false, mod ) )
  {
    mod->error( "Unable to obtain a list of boundary elements for smoothing the boundary." );
    return ofh;
  }

  MatrixHandle interp1(0);  
  boundary_algo->execute( mod, ofh->mesh(), boundary_field_h, interp1, 1 );
  
  const TypeDescription *std = boundary_field_h->get_type_description();
  CompileInfoHandle ci_boundary_smooth = MeshSmootherAlgo::get_compile_info( std, "Quad" );
  Handle<MeshSmootherAlgo> bound_smooth_algo;
  if (!DynamicCompilation::compile( ci_boundary_smooth, bound_smooth_algo, false, mod ) )
  {
    mod->error( "Unable to compile MeshSmoother algorithm for smoothing the boundary." );
    return ofh;
  }

  FieldHandle smoothfield = bound_smooth_algo->execute( mod, boundary_field_h, false, scheme );

  QuadSurfMesh<QuadBilinearLgn<Point> > *smooth_boundary = dynamic_cast<QuadSurfMesh<QuadBilinearLgn<Point> >*>(smoothfield->mesh().get_rep());
  QuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator bound_iter; 
  smooth_boundary->begin( bound_iter );
  QuadSurfMesh<QuadBilinearLgn<Point> >::Node::iterator bound_itere; 
  smooth_boundary->end( bound_itere );
  while( bound_iter != bound_itere )
  {
    QuadSurfMesh<QuadBilinearLgn<Point> >::Node::index_type bi = *bound_iter;
    ++bound_iter;
      
    int size, stride, *cols;
    double *vals;
    interp1->getRowNonzerosNoCopy( bi, size, stride, cols, vals );
      
    Point p;
    smooth_boundary->get_point( p, bi );
    ofield->get_typed_mesh()->set_point( p, *cols );
  }

  return ofh;
}


template <class FIELD>
class MeshSmootherAlgoTri : public MeshSmootherAlgoShared<FIELD>
{
protected:
  virtual FieldHandle compute_boundary( ProgressReporter *reporter,
                                        FieldHandle fieldh,
                                        string scheme );

  virtual bool compute_domain_surface( ProgressReporter *reporter,
                                       FieldHandle fieldh,
                                       FieldHandle &ofieldh );
};


template <class FIELD>
FieldHandle
MeshSmootherAlgoTri<FIELD>::compute_boundary( ProgressReporter *reporter,
                                              FieldHandle fieldh,
                                              string scheme )
{
  reporter->warning( "Currently unable to smooth the boundary of triangle meshes." );
  reporter->remark( "Proceeding with smoothing of the interior elements." );
  return fieldh;
}


template <class FIELD>
bool
MeshSmootherAlgoTri<FIELD>::compute_domain_surface( ProgressReporter *mod,
                                                    FieldHandle fieldh,
                                                    FieldHandle &ofieldh)
{
  ofieldh = fieldh;
  return true;
}


template <class FIELD>
class MeshSmootherAlgoQuad : public MeshSmootherAlgoShared<FIELD>
{
protected:
  virtual FieldHandle compute_boundary( ProgressReporter *reporter,
                                        FieldHandle fieldh,
                                        string scheme );

  virtual bool compute_domain_surface( ProgressReporter *reporter,
                                       FieldHandle fieldh,
                                       FieldHandle &ofieldh );
};


template <class FIELD>
FieldHandle
MeshSmootherAlgoQuad<FIELD>::compute_boundary( ProgressReporter *reporter,
                                               FieldHandle fieldh,
                                               string scheme )
{
  reporter->warning( "Currently unable to smooth the boundary of quadrilateral meshes." );
  reporter->remark( "Proceeding with smoothing of the interior elements." );
  return fieldh;
}


template <class FIELD>
bool
MeshSmootherAlgoQuad<FIELD>::compute_domain_surface( ProgressReporter *mod,
                                                     FieldHandle fieldh,
                                                     FieldHandle &ofieldh )
{
  // The QuadSurfMesh class doesn't currently support a 'snap_to'
  // function, so we'll convert the quads to tris for the domain 
  // functions until the classes can be updated appropriately.
  const TypeDescription *src_td = fieldh->get_type_description();
  CompileInfoHandle qci = QuadToTriAlgo::get_compile_info(src_td);
  Handle<QuadToTriAlgo> qalgo;
  if( !DynamicCompilation::compile(qci, qalgo, mod )) return false;
  if( !qalgo.get_rep() || !qalgo->execute( mod, fieldh, ofieldh ) )
  {
    mod->warning( "QuadToTri conversion failed to copy data." );
    return false;
  }
    
  return true;
}


} // end namespace SCIRun

#endif // MeshSmoother_h
