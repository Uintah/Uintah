#include <Packages/Uintah/CCA/Components/ICE/customInitialize.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Point.h>

using namespace Uintah;
namespace Uintah {
/*_____________________________________________________________________ 
 Function~  customInitialization_problemSetup--
 Purpose~   This function reads in the input parameters for each of the
            customized initialization.  All of the inputs are stuffed into
            the customInitialize_basket.
_____________________________________________________________________*/
void customInitialization_problemSetup( const ProblemSpecP& cfd_ice_ps,
                                        customInitialize_basket* cib)
{
  //__________________________________
  //  search the ICE problem spec for 
  // custom initialization inputs
  ProblemSpecP c_init_ps= cfd_ice_ps->findBlock("customInitialization");
  cib->which = "none";  // default
  
  if(c_init_ps){
    //__________________________________
    // multiple vortices section
    ProblemSpecP vortices_ps= c_init_ps->findBlock("vortices");    
    if(vortices_ps) {
      cib->vortex_inputs = scinew vortices();
      cib->which = "vortices";
      
      for (ProblemSpecP vortex_ps = vortices_ps->findBlock("vortex"); vortex_ps != 0;
                        vortex_ps = vortex_ps->findNextBlock("vortex")) {
        Point origin;
        double strength;
        double radius;
        if(vortex_ps){
          vortex_ps->require("origin",   origin);
          vortex_ps->require("strength", strength);
          vortex_ps->require("radius",   radius);
          cib->vortex_inputs->origin.push_back(origin);
          cib->vortex_inputs->strength.push_back(strength);
          cib->vortex_inputs->radius.push_back(radius);
        }
      }
    }  // multiple vortices
    //__________________________________
    //  method of manufactured solutions 1
    ProblemSpecP mms_ps= c_init_ps->findBlock("manufacturedSolution");
    if(mms_ps) {
      cib->which = "mms_1";
      cib->mms_inputs = scinew mms();
      mms_ps->require("A", cib->mms_inputs->A);
    } 
  }
}
/*_____________________________________________________________________ 
 Function~  customInitialization()--
 Purpose~  overwrite the initialization from the ups file.
_____________________________________________________________________*/ 
void customInitialization(const Patch* patch,
                          CCVariable<double>& rho_CC,
                          CCVariable<double>& temp_CC,
                          CCVariable<Vector>& vel_CC,
                          CCVariable<double>& press_CC,
                          ICEMaterial* ice_matl,
                          const customInitialize_basket* cib)
{
  //__________________________________
  //  multiple vortices
  //See "Boundary Conditions for Direct Simulations of Compressible Viscous
  //     Flows" by Poinsot & LeLe pg 121
  if (cib->which == "vortices"){
    for (int i = 0; i<(int) cib->vortex_inputs->origin.size(); i++) {
      
      Point origin = cib->vortex_inputs->origin[i]; // vortex origin
      double C1 = cib->vortex_inputs->strength[i];  // vortex strength
      double R = cib->vortex_inputs->radius[i];     // vortex radius
      double R_sqr = R * R;
      double p_ref  = 101325;        // assumed reference pressure
     
      for(CellIterator iter=patch->getCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        Point pt = patch->cellPosition(c);

        double x = pt.x() - origin.x();
        double y = pt.y() - origin.y();
        double r = sqrt( x*x + y*y);

        if(r <= R ) {
          double A = exp( -((x*x) + (y*y) )/(2 * R_sqr)); 

          press_CC[c] = p_ref + rho_CC[c] * ((C1*C1)/R_sqr) * A; 

          double uvel = vel_CC[c].x();
          double vvel = vel_CC[c].y();

          vel_CC[c].x(uvel - ((C1 * y)/(rho_CC[c] * R_sqr) ) * A);
          vel_CC[c].y(vvel + ((C1 * x)/(rho_CC[c] * R_sqr) ) * A);
        }
      } 
    }  // loop
  } // vortices
  
  //__________________________________
  // method of manufactured solution 1
  // See:  "A non-trival analytical solution to the 2d incompressible
  //        Navier-Stokes equations" by Randy McDermott
  if(cib->which == "mms_1"){
    double t = 0.0; 
    double A = cib->mms_inputs->A;
    double nu = ice_matl->getViscosity();
    double cv = ice_matl->getSpecificHeat();
    double gamma = ice_matl->getGamma();
    double p_ref = 101325;
    
    for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
      IntVector c = *iter;
      Point pt = patch->cellPosition(c);
      double x = pt.x(); 
      double y = pt.y();
    
      press_CC[c] = p_ref - A*A/4.0 * exp(-4.0*nu*t)
                   *( cos(2.0*(x-t)) + cos(2.0*(y-t)) );

      vel_CC[c].x( 1.0 - A * cos(x-t) * sin(y -t) * exp(-2.0*nu*t));
      vel_CC[c].y( 1.0 + A * sin(x-t) * cos(y -t) * exp(-2.0*nu*t));
      
      // back out temperature from the perfect gas law
      temp_CC[c]= press_CC[c]/ ( (gamma - 1.0) * cv * rho_CC[c] );
    }
  } // mms_1
}
} // end uintah namespace
