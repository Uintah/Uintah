
#ifndef __GEOMETRY_OBJECT2_H__
#define __GEOMETRY_OBJECT2_H__

#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

using namespace SCIRun;

class ICEMaterial;
class GeometryPiece;

/**************************************
       
CLASS
   GeometryObject2
       
   Short description...
       
GENERAL INFORMATION
       
   GeometryObject2.h
       
   John A. Schmidt
   Department of Mechanical Engineering
   University of Utah
       
   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
       
KEYWORDS
   GeometryObject2
       
DESCRIPTION
   Long description...
       
WARNING
       
****************************************/

      class GeometryObject2 {
        
      public:
       //////////
       // Insert Documentation Here:
       GeometryObject2(ICEMaterial* mpm_matl,GeometryPiece* piece,
                     ProblemSpecP&);

       //////////
       // Insert Documentation Here:
        ~GeometryObject2();

         //////////
         // Insert Documentation Here:
         IntVector getNumParticlesPerCell();

        //////////
        // Insert Documentation Here:
        GeometryPiece* getPiece() const {
           return d_piece;
        }

        Vector getInitialVelocity() const {
           return d_initialVel;
        }

        double getInitialTemperature() const {
           return d_initialTemperature;
        }
        
        double getInitialDensity() const {
           return d_initialDensity;
        }
        double getInitialPressure() const {
           return d_initialPressure;
        }

      private:
        GeometryPiece* d_piece;
        IntVector d_resolution;
        Vector d_initialVel;
        double d_initialTemperature,
               d_initialPressure,
               d_initialDensity;
      };
} // End namespace Uintah
      

#endif // __GEOMETRY_OBJECT2_H__

