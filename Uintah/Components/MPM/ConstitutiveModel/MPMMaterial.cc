//  MPMMaterial.cc
//

#include "MPMMaterial.h"
#include "ConstitutiveModel.h"
#include <SCICore/Geometry/IntVector.h>
#include <Uintah/Components/MPM/GeometrySpecification/GeometryPieceFactory.h>
#include <Uintah/Components/MPM/GeometrySpecification/UnionGeometryPiece.h>
#include <Uintah/Components/MPM/GeometrySpecification/GeometryObject.h>
#include <Uintah/Exceptions/ParameterNotFound.h>
#include <iostream>
using namespace std;
using namespace Uintah::Exceptions;
using namespace Uintah::Components;
using namespace SCICore::Geometry;

MPMMaterial::MPMMaterial(ProblemSpecP& ps)
{

   std::string material_type;
   ps->require("material_type", material_type);
   cerr << "material_type is " <<  material_type << endl;

   // Loop through all of the pieces in this geometry object

   int piece_num = 0;
   for (ProblemSpecP geom_obj_ps = ps->findBlock("geom_object");
	geom_obj_ps != 0; 
	geom_obj_ps = geom_obj_ps->findNextBlock("geom_object") ) {

      vector<GeometryPiece*> pieces;
      GeometryPieceFactory::create(geom_obj_ps, pieces);

      GeometryPiece* mainpiece;
      if(pieces.size() == 0){
	 throw ParameterNotFound("No piece specified in geom_object");
      } else if(pieces.size() > 1){
	 mainpiece = new UnionGeometryPiece(pieces);
      } else {
	 mainpiece = pieces[0];
      }

      piece_num++;
      cerr << "piece: " << piece_num << '\n';
      IntVector res;
      geom_obj_ps->require("res",res);
      cerr << piece_num << ": res: " << res << '\n';
      d_geom_objs.push_back(new GeometryObject(mainpiece, res));
   }
   // Constructor

#if 0
   d_cm = ConstitutiveModelFactory::create(ps);
#endif

   ps->require("density",d_density);
   ps->require("toughness",d_toughness);
   ps->require("thermal_conductivity",d_thermal_cond);
   ps->require("specific_heat",d_spec_heat);
}

MPMMaterial::~MPMMaterial()
{
  // Destructor
}

ConstitutiveModel * MPMMaterial::getConstitutiveModel()
{
  // Return the pointer to the constitutive model associated
  // with this material

  return d_cm;
}
