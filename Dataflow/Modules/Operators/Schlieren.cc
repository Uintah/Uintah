#include "Schlieren.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>

//#include <SCICore/Math/Mat.h>


using namespace SCIRun;

namespace Uintah {
 
DECLARE_MAKER(Schlieren)

Schlieren::Schlieren(GuiContext* ctx)
  : Module("Schlieren",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}
  
void Schlieren::execute(void) 
{
  //  tcl_status.set("Calling InPlaneEigenEvaluator!"); 
  in    = (FieldIPort *) get_iport("Scalar Field");
  sfout = (FieldOPort *) get_oport("Scalar Field");

  FieldHandle hTF;
  
  if(!in->get(hTF)){
    std::cerr<<"Schlieren::execute(void) Didn't get a handle\n";
    return;
  } else if ( hTF->get_type_name(1) != "double" ){
    std::cerr<<"Input is not a Scalar field\n";
    return;
  }

  FieldHandle fh = 0;
  if( LatVolField<double> *scalarField1 =
      dynamic_cast<LatVolField<double>*>(hTF.get_rep())) {
    LatVolField<double>  *scalarField2 = 0;  

    scalarField2 = scinew LatVolField<double>(hTF->basis_order());

    computeSchlierenImage(scalarField1); 
    
    fh = scalarField1;
  }
  if( fh.get_rep() != 0 )
    sfout->send(fh);
}

} // end namespace Uintah



