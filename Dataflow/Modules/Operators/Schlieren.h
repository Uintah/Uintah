#ifndef __OPERATORS_Schlieren_H__
#define __OPERATORS_Schlieren_H__

#include "UnaryFieldOperator.h"
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Dataflow/Ports/FieldPort.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <iostream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using std::string;
using std::cerr;
using std::endl;
using namespace SCIRun;

  class Schlieren: public Module, public UnaryFieldOperator {
  public:
    Schlieren(GuiContext* ctx);
    virtual ~Schlieren() {}
    
    virtual void execute(void);
    
  protected:
    template<class ScalarField>       
     void computeSchlierenImage(ScalarField* density);
    
  private:
    //    TCLstring tcl_status;
    GuiInt guiOperation;

    FieldIPort *in;
    FieldOPort *sfout;
    //ScalarFieldOPort *vfout;
  };
  

//__________________________________
template<class ScalarField>       
void  Schlieren::computeSchlierenImage(ScalarField* density)
{

  // only CC data
  ASSERT( density->basis_order() == 0 );
  typename ScalarField::mesh_handle_type sfmh =density->get_typed_mesh();
  typename ScalarField::mesh_type *mesh =density->get_typed_mesh().get_rep();

  typename ScalarField::mesh_type::Cell::iterator v_it;
  typename ScalarField::mesh_type::Cell::iterator v_end;  
  sfmh->begin(v_it);
  sfmh->end(v_end);
  
  BBox mesh_boundary = density->mesh()->get_bounding_box();
  Point min, max;
  min = mesh_boundary.min(); 
  max = mesh_boundary.max();
  LatVolMesh::Cell::index_type min_index, max_index;
  mesh->get_cells(min_index, max_index, BBox(min, max));
  
  cout << " mesh min: " << min << " max " << max << endl;
  cout << " min " << min_index << " max " << max_index << endl;
  cout << " v_begin " << v_it << " v_end " << v_end << endl;

  for( ; v_it != v_end; ++v_it){ 
    double schlieren = 0;    
    LatVolMesh::Cell::index_type c(mesh, v_it.i_+1,  v_it.j_+1, v_it.k_);
   
   
    if ( (v_it.i_ > min_index.i_    && v_it.j_ > min_index.j_    && v_it.k_ > min_index.k_ ) && 
         (v_it.i_ < max_index.i_ -1 && v_it.j_ < max_index.j_-1  && v_it.k_ < max_index.k_ ) ){

      LatVolMesh::Cell::index_type R(mesh, c.i_+1, c.j_,   c.k_);
      LatVolMesh::Cell::index_type L(mesh, c.i_-1, c.j_,   c.k_);
      LatVolMesh::Cell::index_type T(mesh, c.i_,   c.j_+1, c.k_); 
      LatVolMesh::Cell::index_type B(mesh, c.i_,   c.j_-1, c.k_);
     // LatVolMesh::Cell::index_type F(mesh, c.i_,   c.j_,   c.k_+1);
     // LatVolMesh::Cell::index_type BK(mesh,c.i_,   c.j_,   c.k_-1);
      
#if 0  
      cout<< "C " << c << "R " << R << " L " << L << " T " << T << " B " << B  << " F " << F << " BK " << BK << endl;
      cout<< "R " << density->fdata()[R] << " L " << density->fdata()[L]  <<endl;
      cout<< "T " << density->fdata()[T] << " B " << density->fdata()[B]  << endl;
      cout<< "F " << density->fdata()[F] << " BK "<< density->fdata()[BK] << endl;
#endif    
      Vector dx(0.001, 0.001, 0.001);
      double gx=0, gy=0, gz=0;
      
      gx = (density->fdata()[R] - density->fdata()[L])/2.0*dx.x();
      gy = (density->fdata()[T] - density->fdata()[B])/2.0*dx.y();
      gz = 0;
      //gz = (density->fdata()[F] - density->fdata()[BK])/2.0*dx.z();
      schlieren = sqrt(gx*gx + gy*gy + gz*gz);
      
//      Vector gradient(gx,gy,gz);
//    cout << " gradient " << gradient << endl;
   }
   density->fdata()[*v_it] =schlieren;
  }
}

}

#endif // __OPERATORS_Schlieren_H__

