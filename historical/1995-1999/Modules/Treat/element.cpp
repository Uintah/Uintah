#include"element.h"
#include<math.h>

double Element::N[8][8];
double Element::dNds[8][8];
double Element::dNdt[8][8];
double Element::dNdu[8][8];

void Element::makeShape() {

  double gauss_location[2];
  short gauss_pt,i,j,k;
  double s,t,u;
  
  gauss_location[0] = -.57735026918963;
  gauss_location[1] = -gauss_location[0];
  gauss_pt = 0;
  
  for (i=0;i<2;i++) {
    for (j=0;j<2;j++) {
      for (k=0;k<2;k++) {
	
	s = gauss_location[i];
	t = gauss_location[j];
	u = gauss_location[k];
	
	// shape functions evaluated at gauss point
	Element::N[0][gauss_pt]=.125*(1-s)*(1-t)*(1-u);
	Element::N[1][gauss_pt]=.125*(1+s)*(1-t)*(1-u);
	Element::N[2][gauss_pt]=.125*(1+s)*(1+t)*(1-u);
	Element::N[3][gauss_pt]=.125*(1-s)*(1+t)*(1-u);
	Element::N[4][gauss_pt]=.125*(1-s)*(1-t)*(1+u);
	Element::N[5][gauss_pt]=.125*(1+s)*(1-t)*(1+u);
	Element::N[6][gauss_pt]=.125*(1+s)*(1+t)*(1+u);
	Element::N[7][gauss_pt]=.125*(1-s)*(1+t)*(1+u);
	
	// dN/ds evaluated at gauss point
	Element::dNds[0][gauss_pt]=-.125*(1-t)*(1-u);
	Element::dNds[1][gauss_pt]= .125*(1-t)*(1-u);
	Element::dNds[2][gauss_pt]= .125*(1+t)*(1-u);
	Element::dNds[3][gauss_pt]=-.125*(1+t)*(1-u);
	Element::dNds[4][gauss_pt]=-.125*(1-t)*(1+u);
	Element::dNds[5][gauss_pt]= .125*(1-t)*(1+u);
	Element::dNds[6][gauss_pt]= .125*(1+t)*(1+u);
	Element::dNds[7][gauss_pt]=-.125*(1+t)*(1+u);
	
	// dN/dt evaluated at gauss point
	Element::dNdt[0][gauss_pt]=-.125*(1-s)*(1-u);
	Element::dNdt[1][gauss_pt]=-.125*(1+s)*(1-u);
	Element::dNdt[2][gauss_pt]= .125*(1+s)*(1-u);
	Element::dNdt[3][gauss_pt]= .125*(1-s)*(1-u);
	Element::dNdt[4][gauss_pt]=-.125*(1-s)*(1+u);
	Element::dNdt[5][gauss_pt]=-.125*(1+s)*(1+u);
	Element::dNdt[6][gauss_pt]= .125*(1+s)*(1+u);
	Element::dNdt[7][gauss_pt]= .125*(1-s)*(1+u);
	
	// dN/du evaluated at gauss point
	Element::dNdu[0][gauss_pt]=-.125*(1-s)*(1-t);
	Element::dNdu[1][gauss_pt]=-.125*(1+s)*(1-t);
	Element::dNdu[2][gauss_pt]=-.125*(1+s)*(1+t);
	Element::dNdu[3][gauss_pt]=-.125*(1-s)*(1+t);
	Element::dNdu[4][gauss_pt]= .125*(1-s)*(1-t);
	Element::dNdu[5][gauss_pt]= .125*(1+s)*(1-t);
	Element::dNdu[6][gauss_pt]= .125*(1+s)*(1+t);
	Element::dNdu[7][gauss_pt]= .125*(1-s)*(1+t);
	
	gauss_pt++;  // gauss point will count up to 8 points
	
      }
    }
  }
 
  
} // end of method makeShape


void Element::makeStiff() {


  /**

     node points                       gauss points
                                            y
          y                                 ^
          |                                 |
          |                                 |
          4------3                          2---|----4
         /|     /|                         /|   |    |
        / |    / |                        / |       /|
       /  1---/--2 ---x                  /  |      / |
     8------7   /                       /   1-----/--3
     | /    |  /                       6--------8   ------->x    
     |/     | /                        |        |   /
     5------6                          |    /   |  /
    /                                  |   /    | /
   /                                   5--/-----7
  z                                   /
                                     /
                                    z
     
     shape funtions at each gauss point, shape(shape function,gauss point).
     the signs coorespond to the coordinates of the node for that shape
     function : (x,y,z).  the gauss points are located as shown above
     
     8 total gauss points per element
     
     the real space is (x,y,z)
     and the mapped (square of side length 2) space is (s,t,u)
     the shape functions above are evaluated in the (s,t,u) space.
     the jacobian found here calculates the terms
     
     | dx/ds  dy/ds  dz/ds |
     | dx/dt  dy/dt  dz/dt |
     | dx/du  dy/du  dz/du |
     
     found from the relation:
     
     dx   |dN1 dN2 dN3     dN8 |                   t
     -- = |--- --- --- ... --  |  [x1 x2 x3 ... x8]
     ds   |ds  ds  ds      ds  |
     
     */   

  
  double J[3][3];  // Jacobian Matrix
  double invJ[3][3]; // Inverse of the Jacobian
  double det; // determinate of the Jacobian
  double B[3][8]; // dNd[x]
  double hzeta[3];
  double heta[3];
  double hpsi[3];  
  double u=0,v=0,w=0,uabs2;
  double h,ha,hb,hc;
  double alphax,cof,gam,tanh;
  double wx,wy,wz,udppvdp,pg,vis1,vis2,vis3,cx,cy,cz,conv,w1;
  
//     element mesh length vectors for petrov-galerkin
     
  hzeta[0]=(double)((node[6]->getX()-node[7]->getX()+node[1]->getX()-node[0]->getX()+node[2]->getX()-node[3]->getX()+node[5]->getX()-node[4]->getX())*.25);
  hzeta[1]=(double)((node[6]->getY()-node[7]->getY()+node[1]->getY()-node[0]->getY()+node[2]->getY()-node[3]->getY()+node[5]->getY()-node[4]->getY())*.25);
  hzeta[2]=(double)((node[6]->getZ()-node[7]->getZ()+node[1]->getZ()-node[0]->getZ()+node[2]->getZ()-node[3]->getZ()+node[5]->getZ()-node[4]->getZ())*.25);
  
  heta[0]=(double)((node[6]->getX()-node[5]->getX()+node[2]->getX()-node[1]->getX()+node[3]->getX()-node[0]->getX()+node[7]->getX()-node[4]->getX())*.25);
  heta[1]=(double)((node[6]->getY()-node[5]->getY()+node[2]->getY()-node[1]->getY()+node[3]->getY()-node[0]->getY()+node[7]->getY()-node[4]->getY())*.25);
  heta[2]=(double)((node[6]->getZ()-node[5]->getZ()+node[2]->getZ()-node[1]->getZ()+node[3]->getZ()-node[0]->getZ()+node[7]->getZ()-node[4]->getZ())*.25);
  
  hpsi[0]=(double)((node[5]->getX()-node[1]->getX()+node[6]->getX()-node[2]->getX()+node[4]->getX()-node[0]->getX()+node[7]->getX()-node[3]->getX())*.25);
  hpsi[1]=(double)((node[5]->getY()-node[1]->getY()+node[6]->getY()-node[2]->getY()+node[4]->getY()-node[0]->getY()+node[7]->getY()-node[3]->getY())*.25);
  hpsi[2]=(double)((node[5]->getZ()-node[1]->getZ()+node[6]->getZ()-node[2]->getZ()+node[4]->getZ()-node[0]->getZ()+node[7]->getZ()-node[3]->getZ())*.25);
  

  for (int gauss=0;gauss<8;gauss++) { // integrate by looping over gauss pts.

    // find the jacobian matrix "J"

    // clear it out first

    int i;
    for (i=0;i<3;i++){
      for (int j=0;j<3;j++) {
	J[i][j] = 0;
      }
    }
 
    for (i=0;i<8;i++) { // loop through nodes
      J[0][0] += dNds[i][gauss]*node[i]->getX();
      J[0][1] += dNds[i][gauss]*node[i]->getY();
      J[0][2] += dNds[i][gauss]*node[i]->getZ();
      J[1][0] += dNdt[i][gauss]*node[i]->getX();
      J[1][1] += dNdt[i][gauss]*node[i]->getY();
      J[1][2] += dNdt[i][gauss]*node[i]->getZ();
      J[2][0] += dNdu[i][gauss]*node[i]->getX();
      J[2][1] += dNdu[i][gauss]*node[i]->getY();
      J[2][2] += dNdu[i][gauss]*node[i]->getZ();
    }
    
    // find the determinate of the Jacobian (scaler)

    det = (J[0][0]*J[1][1]*J[2][2] + J[0][1]*J[1][2]*J[2][0] + 
	   J[0][2]*J[1][0]*J[2][1] - J[2][0]*J[1][1]*J[0][2] - 
	   J[2][1]*J[1][2]*J[0][0] - J[2][2]*J[1][0]*J[0][1]);
    
    volume += det;

    // find the inverse of the Jacobian (3x3 matrix)

    invJ[0][0]=(J[1][1]*J[2][2]-J[2][1]*J[1][2])/det;
    invJ[0][1]=(J[2][1]*J[0][2]-J[0][1]*J[2][2])/det;
    invJ[0][2]=(J[0][1]*J[1][2]-J[1][1]*J[0][2])/det;
    invJ[1][0]=(J[2][0]*J[1][2]-J[1][0]*J[2][2])/det;
    invJ[1][1]=(J[0][0]*J[2][2]-J[2][0]*J[0][2])/det;
    invJ[1][2]=(J[1][0]*J[0][2]-J[0][0]*J[1][2])/det;
    invJ[2][0]=(J[1][0]*J[2][1]-J[2][0]*J[1][1])/det;
    invJ[2][1]=(J[2][0]*J[0][1]-J[0][0]*J[2][1])/det;
    invJ[2][2]=(J[0][0]*J[1][1]-J[1][0]*J[0][1])/det;
    
    // find B matrix (3x8 matrix)
    
    for (i=0;i<3;i++) { // loop through rows
      for (int j=0;j<8;j++) { // loop through columns
	B[i][j] = (invJ[i][0]*dNds[j][gauss] + 
		   invJ[i][1]*dNdt[j][gauss] +
		   invJ[i][2]*dNdu[j][gauss]);
      }
    }

//     find the velocity at the gauss point

    u=0;v=0;w=0;
    for (i=0;i<8;i++) { // loop over nodes in element
      u=u+node[i]->getVX()*N[i][gauss];
      v=v+node[i]->getVY()*N[i][gauss];
      w=w+node[i]->getVZ()*N[i][gauss];
    }

    uabs2 = u*u + v*v + w*w;
    ha = u*hzeta[0]+v*hzeta[1]+w*hzeta[2];
    hb = u*heta[0]+v*heta[1]+w*heta[2];
    hc = u*hpsi[0]+v*hpsi[1]+w*hpsi[2];
    h = fabs(ha) + fabs(hb) + fabs(hc);
    
    //     petrov galerkin coefficient for the energy equation
    //     
    //     h=||v||*h
    //     gam=h/alpha
    //     cof=alphax*h/(2||v||)
  
    if (uabs2 > 1.e-20) {
      gam = (double)(h/alpha);
      tanh = (double)((exp(gam/2)-exp(-gam/2))/
	(exp(gam/2)+exp(-gam/2)));
      alphax = (double)(1/(tanh)-2/gam);
      cof = (double)(alphax*h*(.5/uabs2));
      // cof=alphax*h*.333333/uabs2
    } else {
      cof = 0;
    }
    
    
    for (i=0;i<8;i++) { // loop through weighting functions
      wx=B[0][i]*fabs(det)*alpha; 
      wy=B[1][i]*fabs(det)*alpha;
      wz=B[2][i]*fabs(det)*alpha;
      
      udppvdp = u*B[0][i] + v*B[1][i] + w*B[2][i];
      pg = (N[i][gauss] + cof*udppvdp)*fabs(det);

      for (int j=0;j<8;j++) { // loop through shape functions
	vis1=wx*B[0][j]; // thermal stiffness
	vis2=wy*B[1][j];
	vis3=wz*B[2][j];
	cx=B[0][j]*u;
	cy=B[1][j]*v;
	cz=B[2][j]*w;
	conv=pg*(cx+cy+cz);
	w1=pg*N[j][gauss];
	thermStiff[i][j]=thermStiff[i][j] + vis1 + vis2 + vis3 + w1*perf +conv;
	massStiff[i][j]=massStiff[i][j] + w1;

      } // end loop over shape functions

    } // end loop over weighting functions

  } // end of loop over gauss points
  

} // end of method makeStiff      
  
    








