#ifndef __MATERIAL_H__
#define __MATERIAL_H__

#include <Uintah/Interface/DataWarehouseP.h>
#include "ConstitutiveModel.h"

/**************************************

CLASS
   Material

   Short description...

GENERAL INFORMATION

   Material.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   Copyright (C) 2000 SCI Group

KEYWORDS
   Material

DESCRIPTION
   Long description...

WARNING

****************************************/

//using Uintah::Grid::Region;
//using Uintah::Interface::DataWarehouseP;


class Material {
public:
  Material() {}

  ~Material() {}


private:

  Material(const Material &mat);
  Material& operator=(const Material &mat);

};

#endif // __MATERIAL_H__

// $Log$
// Revision 1.1  2000/03/24 00:45:43  guilkey
// Added MPMMaterial class, as well as a skeleton Material class, from
// which MPMMaterial is inherited.  The Material class will be filled in
// as it's mission becomes better identified.
//

