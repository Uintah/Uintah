#include    <Core/Geometry/Point.h>
#include    <Core/Geometry/Vector.h>
#include    <Packages/rtrt/Core/Ray.h>
#include <iostream>

namespace rtrt {
// return non-unit-length normal
Vector GradientCell(const Point& pmin, const Point& pmax,
            const Point& p, float rho[2][2][2]) {

double x = p.x();
double y = p.y();
double z = p.z();

double x_0 = pmin.x();
double y_0 = pmin.y();
double z_0 = pmin.z();

double x_1 = pmax.x();
double y_1 = pmax.y();
double z_1 = pmax.z();

double N_x, N_y, N_z;

N_x  =   - (y_1-y)*(z_1-z)*rho[0][0][0] 
         + (y_1-y)*(z_1-z)*rho[1][0][0]
         - (y-y_0)*(z_1-z)*rho[0][1][0]
         - (y_1-y)*(z-z_0)*rho[0][0][1]
         + (y-y_0)*(z_1-z)*rho[1][1][0] 
         + (y_1-y)*(z-z_0)*rho[1][0][1]
         - (y-y_0)*(z-z_0)*rho[0][1][1]
         + (y-y_0)*(z-z_0)*rho[1][1][1];

N_y  =   - (x_1-x)*(z_1-z)*rho[0][0][0]
         - (x-x_0)*(z_1-z)*rho[1][0][0]
         + (x_1-x)*(z_1-z)*rho[0][1][0]
         - (x_1-x)*(z-z_0)*rho[0][0][1]
         + (x-x_0)*(z_1-z)*rho[1][1][0]
         - (x-x_0)*(z-z_0)*rho[1][0][1]
         + (x_1-x)*(z-z_0)*rho[0][1][1]
         + (x-x_0)*(z-z_0)*rho[1][1][1];

N_z =    - (x_1-x)*(y_1-y)*rho[0][0][0]
         - (x-x_0)*(y_1-y)*rho[1][0][0]
         - (x_1-x)*(y-y_0)*rho[0][1][0]
         + (x_1-x)*(y_1-y)*rho[0][0][1]
         - (x-x_0)*(y-y_0)*rho[1][1][0]
         + (x-x_0)*(y_1-y)*rho[1][0][1]
         + (x_1-x)*(y-y_0)*rho[0][1][1]
         + (x-x_0)*(y-y_0)*rho[1][1][1];
    
return Vector(N_x, N_y, N_z);

}

// return non-unit-length normal
Vector GradientCell(const Point& pmin, const Point& pmax,
                    const Point& p, float rho[2][2]) {

double x = p.x();
double y = p.y();

double x_0 = pmin.x();
double y_0 = pmin.y();

double x_1 = pmax.x();
double y_1 = pmax.y();

double N_x, N_y, N_z;

N_x  =     (y_1-y)*(rho[0][1]-rho[1][1])
         + (y-y_0)*(rho[0][0]-rho[1][0]);

N_y  =     (x_1-x)*(rho[1][0]-rho[1][1])
         + (x-x_0)*(rho[0][0]-rho[0][1]);

N_z =    (y_1-y_0)*(x_1-x_0);
    
return Vector(N_x, N_y, N_z);
}

} // end namespace rtrt
