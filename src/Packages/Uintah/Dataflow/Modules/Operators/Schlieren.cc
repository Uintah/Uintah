#include "Schlieren.h"
#include <math.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/Disclosure/TypeUtils.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>

using namespace SCIRun;

namespace Uintah {
 
DECLARE_MAKER(Schlieren)

Schlieren::Schlieren(GuiContext* ctx)
  : Module("Schlieren",ctx,Source, "Operators", "Uintah"),
    guiOperation(ctx->subVar("operation"))
{
}


//__________________________________  
void Schlieren::execute(void) 
{
  in    = (FieldIPort *) get_iport("Scalar Field");
  sfout = (FieldOPort *) get_oport("Scalar Field");
  FieldHandle hTF;
  
  // bullet proofing
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
    
    LatVolField<double>  *output = 0;  
    output = scinew LatVolField<double>(hTF->basis_order());
    
    // compute the image
    computeSchlierenImage(scalarField1,output); 
    
    fh = output;
  }
  if( fh.get_rep() != 0 )
    sfout->send(fh);
}

} // end namespace Uintah



