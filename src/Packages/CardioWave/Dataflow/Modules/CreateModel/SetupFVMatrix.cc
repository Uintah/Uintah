/*
 *  SetupFVMatrix.cc:
 *
 *   Written by:
 *   Joe Tranquillo
 *   Duke University 
 *   Biomedical Engineering Department
 *   August 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

extern "C" {
#include <Packages/CardioWave/Core/Algorithms/Vulcan.h>
}

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE SetupFVMatrix : public Module {
	GuiDouble	sigx_;
	GuiDouble	sigy_;
	GuiDouble	sigz_;
	GuiString	sprfile_;
	GuiString	volumefile_;

public:
  SetupFVMatrix(const string& id);
  virtual ~SetupFVMatrix();
  virtual void execute();
};

extern "C" CardioWaveSHARE Module* make_SetupFVMatrix(const string& id) {
  return scinew SetupFVMatrix(id);
}

SetupFVMatrix::SetupFVMatrix(const string& id)
  : Module("SetupFVMatrix", id, Source, "CreateModel", "CardioWave"),
    sigx_("sigx", id, this),
    sigy_("sigy", id, this),
    sigz_("sigz", id, this),
    sprfile_("sprfile", id, this),
    volumefile_("volumefile", id, this)
{
}

SetupFVMatrix::~SetupFVMatrix(){
}

void SetupFVMatrix::execute(){
	double sigx = sigx_.get();
	double sigy = sigy_.get();
	double sigz = sigz_.get();
	string sprfile = sprfile_.get();
	string volumefile = volumefile_.get();

//	FILE HANDEL IN

	MESH * mesh;

	// fill in mesh from SCIRun field

/*		for(k=0; k<=zdim-1; k++){
		for(j=0; j<=ydim-1; j++){
		for(i=0; i<=xdim-1; i++){
			nodenumber = i+j*xdim+k*xdim*ydim;
			(*mesh)->vtx[nodenumber].x = i*dx;
			(*mesh)->vtx[nodenumber].y = j*dy;
			(*mesh)->vtx[nodenumber].z = k*dz;			
		}
		}
		}

		for(k=0; k<=zdim-2; k++){
		for(j=0; j<=ydim-2; j++){
		for(i=0; i<=xdim-2; i++){
			(*mesh)->elements[numelem].vtx[0] = i+j*xdim+k*xdim*ydim;
			(*mesh)->elements[numelem].vtx[1] = (i+1)+j*xdim+k*xdim*ydim;
			(*mesh)->elements[numelem].vtx[2] = (i+1)+(j+1)*xdim+k*xdim*ydim;
			(*mesh)->elements[numelem].vtx[3] = i+(j+1)*xdim+k*xdim*ydim;
			(*mesh)->elements[numelem].vtx[4] = i+j*xdim+(k+1)*xdim*ydim;
			(*mesh)->elements[numelem].vtx[5] = (i+1)+j*xdim+(k+1)*xdim*ydim;
			(*mesh)->elements[numelem].vtx[6] = (i+1)+(j+1)*xdim+(k+1)*xdim*ydim;
			(*mesh)->elements[numelem].vtx[7] = i+(j+1)*xdim+(k+1)*xdim*ydim;
			numelem++;
		}
		}
		}

		for(i=0; i<3; i++){
			for(j=0; j<3; j++){
				U[i][j] = 0.0;
				UT[i][j] = 0.0;
				sigma[i][j] = 0.0;
				D[i][j] = 0.0;
				temp[i][j] = 0.0;	
			}
		}


		printf("Enter the relative conductivity scale [longitudinal transverse sheet] in mS/cm: ");
		scanf("%f %f %f", &sigma[0][0], &sigma[1][1], &sigma[2][2]);
		printf("Enter the longitudinal unit vector [x y z]: ");
		scanf("%f %f %f", &U[0][0], &U[0][1], &U[0][2]);
		printf("Enter the transverse unit vector [x y z]: ");
		scanf("%f %f %f", &U[1][0], &U[1][1], &U[1][2]);


		dot_result = U[0][0]*U[1][0]+U[0][1]*U[1][1]+U[0][2]*U[1][2];
		if(dot_result>0.0001){
			printf("\n\nUnit Vectors not orthogonal!\n");
			exit(-1);
		}
//		 Cross Product 
		U[2][0]=U[0][1]*U[1][2]-U[0][2]*U[1][1]; 
		U[2][1]=U[0][2]*U[1][0]-U[0][0]*U[1][2];
		U[2][2]=U[0][0]*U[1][1]-U[0][1]*U[1][0];
		printf("The sheet unit vector is [%2.2f %2.2f %2.2f].\n",U[2][0],U[2][1],U[2][2]);

//		 U Transpose 
		UT[0][0] = U[0][0];
		UT[0][1] = U[1][0];
		UT[0][2] = U[2][0];
		UT[1][0] = U[0][1];
		UT[1][1] = U[1][1];
		UT[1][2] = U[2][1];
		UT[2][0] = U[0][2];
		UT[2][1] = U[1][2];
		UT[2][2] = U[2][2];

		for(i=0; i<3; i++){
			for(j=0; j<3; j++){
				for(k=0; k<3; k++){
					temp[i][j]=temp[i][j]+U[i][k]*sigma[k][j]; 
				}
			}
		}

		for(i=0; i<3; i++){
			for(j=0; j<3; j++){
				for(k=0; k<3; k++){
					D[i][j]=D[i][j]+temp[i][k]*UT[k][j]; 
				}
			}
		}

		for(i=0;i<mesh->numvtx;i++)
		{
			mesh->vtx[i].sxx=D[0][0];
			mesh->vtx[i].sxy=D[0][1];
			mesh->vtx[i].sxz=D[0][2];			
			mesh->vtx[i].syy=D[1][1];
			mesh->vtx[i].syz=D[1][2];
			mesh->vtx[i].szz=D[2][2];
		}
		
*/
	
	compute_volumes(mesh);
	compute_matrix(mesh);
}

} // End namespace CardioWave


