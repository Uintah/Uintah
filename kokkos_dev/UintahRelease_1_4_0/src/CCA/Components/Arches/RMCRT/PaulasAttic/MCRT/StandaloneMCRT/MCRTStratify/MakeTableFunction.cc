/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


// Making two tables:
// VolTable ( including the real or virtual surfaces flag )
// VolNeighbor ( neighboring table )


// dont want so many parameters, can pass by object
// and enclose all these variables in the object

// making tables is separated from main function
// and this way, this table-making is not general any more,
// it needs to be modified according to the geometry

// how do i calculate the area of cylinder surface on cells
// now do the simulation on the rectangular cylinder

// VolTable:
// z1, z2, y1, y2, x1, x2, VolIndex,
// TopSurfaceIndex, BottomSurfaceIndex,
// FrontSurfaceIndex, BackSurfaceIndex,
// LeftSurfaceIndex, RightSurfaceIndex

#include "MakeTableFunction.h"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <string>
#include <sstream>
#include <cmath>

using namespace std;

// these two tables function can be achieved by Template

/*
template <class ArrayType>
void MakeTableFunction::singleArrayTable(const ArrayType *Array,
					 const int &ArraySize,
					 const int &No,
					 char *_argv){

  ofstream out( _argv);

  for ( int i = 0; i < ArraySize; i ++ ) {    
    out << setw(13) << setprecision(5) << Array[i];
    if ( ( (i+1) % No ) == 0 )
      out << endl;
  }
  
   out.close();
}
*/



void MakeTableFunction::singleArrayTable(const double *Array,
					 const int &ArraySize,
					 const int &No, char * _argv){
  ofstream out( _argv);

  for ( int i = 0; i < ArraySize; i ++ ) {    
    out << setw(13) << setprecision(5) << Array[i];
    if ( ( (i+1) % No ) == 0 )
      out << endl;
  }
  
   out.close();
}



void MakeTableFunction::singleIntArrayTable(const int *Array,
					    const int &ArraySize,
					    const int &No, char *_argv){
  ofstream out( _argv);

  for ( int i = 0; i < ArraySize; i ++ ) {    
    out << setw(13) << setprecision(5) << Array[i];
    if ( ( (i+1) % No ) == 0 )
      out << endl;
  }
  
   out.close();
}




void MakeTableFunction::coupleTwoArrayTable(const int &arraySize,
					    const double *array1,
					    const double *array2,
					    const int &changelineNo,
					    char *_argv){
  
  ofstream out( _argv);
  
  int j = 0;
  
  for( int i = 0; i < arraySize; i ++ ) {
    
    out << setw(13) << setprecision(5) << i; // counter
    out << setw(13) << setprecision(5) << array1[i];
    
    do{
      out << setw(13) << setprecision(5) << array2[j];
      j++;
      
    }while( ( j % changelineNo ) != 0 );
    
    out << endl;
   
  }

  
   out.close();  


}



void MakeTableFunction::twoArrayTable(const int  &arraySize,
				      const double *array1,
				      const double *array2,
				      char *_argv) {

  ofstream out(_argv);

  for ( int arrayCounter = 0; arrayCounter < arraySize; arrayCounter ++ ) {
    
      out << setw(16) << setprecision(5) << arrayCounter; // VolIndex
      out << setw(16) << setprecision(5) << array1[arrayCounter]; // temp
      out << setw(16) << setprecision(5) << array2[arrayCounter]; // absorb coeff
      out << endl;

  }

  out.close();
}




void MakeTableFunction::threeArrayTable(const int  &arraySize,
					const double *array1,
					const double *array2,
					const double *array3,
					char *_argv) {

  ofstream out(_argv);

  for ( int arrayCounter = 0; arrayCounter < arraySize; arrayCounter ++ ) {
    
      out << setw(16) << setprecision(5) << arrayCounter; 
      out << setw(16) << setprecision(5) << array1[arrayCounter]; 
      out << setw(16) << setprecision(5) << array2[arrayCounter]; 
      out << setw(16) << setprecision(5) << array3[arrayCounter];    
      out << endl;

  }

  out.close();
}



void MakeTableFunction::fourArrayTable(const int  &arraySize,
				       const double *array1,
				       const double *array2,
				       const double *array3,
				       const double *array4,
				       char *_argv) {

  ofstream out(_argv);

  for ( int arrayCounter = 0; arrayCounter < arraySize; arrayCounter ++ ) {
    
      out << setw(16) << setprecision(5) << arrayCounter; 
      out << setw(16) << setprecision(5) << array1[arrayCounter]; 
      out << setw(16) << setprecision(5) << array2[arrayCounter]; 
      out << setw(16) << setprecision(5) << array3[arrayCounter];
      out << setw(16) << setprecision(5) << array4[arrayCounter];      
      out << endl;

  }

  out.close();
}





void MakeTableFunction::q_top(const double *Array,
			      const int &xno, const int &yno,
			      char *_argv){
  ofstream out( _argv);
  int index;
  index = xno * yno;
  
  for ( int i = 0; i < index; i ++) {
    out << setw(13) << setprecision(6) << Array[i];
    if ( ( (i+1) % xno ) == 0 )
      out << endl;
  }
  out.close();
  
}



void MakeTableFunction::q_bottom(const double *Array,
				 const int &xno, const int &yno,
				 char *_argv){
  ofstream out( _argv);
  int index;
  index = 2 * xno * yno;
  
  for ( int i = xno * yno; i < index; i ++) {
    out << setw(13) << setprecision(6) << Array[i];
    if ( ( (i+1) % xno ) == 0 )
      out << endl;
  }
  out.close();
  
}


void MakeTableFunction::q_Q(const double *q_surface, const double *qdiv,
			    const int &VolElementNo, const int &totalElementNo, char *_argv){
  
  ofstream out( _argv);

  for ( int i = 0; i < VolElementNo; i ++ ) {
    if ( ( (i+1) % 6 ) == 0 )
      out << endl;
    
    out << setw(10) << setprecision(6) << qdiv[i];
  }

  for ( int i = VolElementNo; i < totalElementNo; i ++) {
    if ( ( i % 6 ) == 0 )
      out << endl;

    out << setw(10) << setprecision(6) << q_surface[i-VolElementNo];
  }
  
   out.close();

}

		     



void MakeTableFunction::VolProperty(  const int  &VolElementNo,
				      const double *T_Vol,
				      const double *kl_Vol,
				      const double *scatter_Vol,
				      const double *emiss_Vol,
				      char *_argv) {

  ofstream out(_argv);

  for ( int VolIndex = 0; VolIndex < VolElementNo; VolIndex ++ ) {
    
      out << setw(16) << setprecision(5) << VolIndex; // VolIndex
      out << setw(16) << setprecision(5) << T_Vol[VolIndex]; // temp
      out << setw(16) << setprecision(5) << kl_Vol[VolIndex]; // absorb coeff
      out << setw(16) << setprecision(5) << scatter_Vol[VolIndex]; // scatter coeff
      out << setw(16) << setprecision(5) << emiss_Vol[VolIndex]; // emiss coeff
      out << endl;

  }

  out.close();
}



void MakeTableFunction::VolProperty_Array(  const int  &VolElementNo,
					    const double *T_Vol,
					    const double *kl_Vol,
					    const double *scatter_Vol,
					    const double *emiss_Vol,
					    double *VolArray){

  int k = 0;
  for ( int i = 0; i < VolElementNo; i ++ ) {

      VolArray[k] = i;
      VolArray[++k] = T_Vol[i];
      VolArray[++k] = kl_Vol[i];
      VolArray[++k] = scatter_Vol[i];
      VolArray[++k] = emiss_Vol[i];
      k++;
    
  }
  
}





void MakeTableFunction::RealSurfaceProperty(  int  TopStartNo,
					      int  surfaceElementNo,
					      const double *T_surface,
					      const double *absorb_surface,
					      const double *rs_surface,
					      const double *rd_surface,
					      const double *emiss_surface,
					      char *_argv){

  ofstream out(_argv);
  int surfaceIndex = TopStartNo;
  
  for ( int i = 0; i < surfaceElementNo; i ++ ) {
    
      out << setw(16) << setprecision(5) << surfaceIndex; // surfaceIndex
      out << setw(16) << setprecision(5) << T_surface[i]; // temp
      out << setw(16) << setprecision(5) << absorb_surface[i]; // absorb coeff
      out << setw(16) << setprecision(5) << rs_surface[i]; // specular reflect coeff
      out << setw(16) << setprecision(5) << rd_surface[i]; // diff reflect coeff
      out << setw(16) << setprecision(5) << emiss_surface[i]; // emiss coeff
      out << endl;
      surfaceIndex ++;
      
  }

  out.close();
}




void MakeTableFunction::RealSurfaceProperty_Array(  int TopStartNo,
						    int surfaceElementNo,
						    const double *T_surface,
						    const double *absorb_surface,
						    const double *rs_surface,
						    const double *rd_surface,
						    const double *emiss_surface,
						    double *SurfaceArray){


  int surfaceIndex = TopStartNo;
  int k = 0;
  for ( int i = 0; i < surfaceElementNo; i ++ ) {
    
      SurfaceArray[k] = surfaceIndex; // surfaceIndex
      SurfaceArray[++k] = T_surface[i]; // temp
      SurfaceArray[++k] = absorb_surface[i]; // absorb coeff
      SurfaceArray[++k] = rs_surface[i]; // specular reflect coeff
      SurfaceArray[++k] = rd_surface[i]; // diff reflect coeff
      SurfaceArray[++k] = emiss_surface[i]; // emiss coeff
     
      surfaceIndex ++;
      k++; 
  }


}




void MakeTableFunction::vtkSurfaceTableMake( const char *_argv, const int &xnop,
					     const int &ynop, const int &znop,
					     const double *X, const double *Y,
					     const double *Z,
					     const int &surfaceElementNo,
					     const double *q_surface,
					     const double *Q_surface){
  
  ofstream out( _argv );// surface element table
  int pointNo;
  
  //vtk format
  //-------------------------------------------------------------
  //Header
  out <<"# vtk DataFile Version 2.0" << endl;
  out <<"Three-D cube surface heat flux and Q display"<<endl;
  out <<"ASCII"<<endl;
  out << "DATASET UNSTRUCTURED_GRID"<<endl;
  out << endl;
  
  //----------------------------------------------------------------
  //PointIndex--xyz coordinates for each point
  pointNo = 2 * ( xnop * ynop + xnop * (znop - 2 ) + ( ynop -2 ) * ( znop-2) ); 
  out << "POINTS " << pointNo << " FLOAT " << endl;

  // top->bottom->front->back->left->right
  // from left to right, back to front, top to bottom
  
  // top surface
  for ( int i = 0; i < ynop; i ++ ) {
    for ( int j = 0; j < xnop; j ++ )
      out << " " << X[j] << " " << Y[i] << " " << Z[0] << endl;
  }

  // bottom surface
  for ( int i = 0; i < ynop; i ++ ) {
    for ( int j = 0; j < xnop; j ++ )
      out << " " << X[j] << " " << Y[i] << " " << Z[znop-1] << endl;
  }

  // front surface
  for ( int i = 1; i < znop-1; i ++ ) {
    for ( int j = 0; j < xnop; j ++ )
      out << " " << X[j] << " " << Y[ynop-1] << " " << Z[i] << endl;
  }  

  // back surface
  for ( int i = 1; i < znop-1; i ++ ) {
    for ( int j = 0; j < xnop; j ++ )
      out << " " << X[j] << " " << Y[0] << " " << Z[i] << endl;
  }

  // left surface
  for ( int i = 1; i < znop-1; i ++ ) {
    for ( int j = 1; j < ynop-1; j ++ )
      out << " " << X[0] << " " << Y[j] << " " << Z[i] << endl;
  }

  // right surface
  for ( int i = 1; i < znop-1; i ++ ) {
    for ( int j = 1; j < ynop-1; j ++ )
      out << " " << X[xnop-1] << " " << Y[j] << " " << Z[i] << endl;
  }  

  out << endl;

  //-------------------------------------------------------------
  //CellIndex
  out << "CELLS " << surfaceElementNo << " " << 5 * surfaceElementNo << endl;

  // generate 6 matrices for 6 surfaces' cell index

  int topcell[ynop][xnop], bottomcell[ynop][xnop];
  int frontcell[znop][xnop], backcell[znop][xnop];
  int leftcell[znop][ynop], rightcell[znop][ynop];
  int start_index = 0;

  // ------ topcell -------
  for ( int i = 0; i < ynop; i ++ ){
    for ( int j = 0; j < xnop; j ++ ){
      topcell[i][j] = start_index;
      start_index ++;
    }
  }

  // ------ bottomcell -------
  for ( int i = 0; i < ynop; i ++ ){
    for ( int j = 0; j < xnop; j ++ ){
      bottomcell[i][j] = start_index;
      start_index ++;
    }
  }

  // ------ frontcell --------
  // first row of frontcell
  for ( int j = 0; j < xnop; j ++ )
    frontcell[0][j] = topcell[ynop-1][j];
  
  for ( int i = 1; i < znop - 1 ; i ++ ){
    for ( int j = 0; j < xnop; j ++ ){
      frontcell[i][j] = start_index;
      start_index ++;
    }
  }

  // last row of frontcell
  for ( int j = 0; j < xnop; j ++ )
    frontcell[znop-1][j] = bottomcell[ynop-1][j];


  
  // ------- backcell ---------
  // first row of backcell
  for ( int j = 0; j < xnop; j ++ )
    backcell[0][j] = topcell[0][j];
  
  for ( int i = 1; i < znop - 1 ; i ++ ){
    for ( int j = 0; j < xnop; j ++ ){
      backcell[i][j] = start_index;
      start_index ++;
    }
  }

  // last row of backcell
  for ( int j = 0; j < xnop; j ++ )
    backcell[znop-1][j] = bottomcell[0][j];


  // ------ leftcell ----------
  // first row and last row of leftcell
  for ( int j = 0; j < ynop; j ++ ){
    leftcell[0][j] = topcell[j][0];
    leftcell[znop-1][j] = bottomcell[j][0];
  }

  // first colume and last colume of leftcell
  for ( int i = 0; i < znop; i ++ ) {
    leftcell[i][0] = backcell[i][0];
    leftcell[i][ynop-1] = frontcell[i][0];
  }
  
  for ( int i = 1; i < znop - 1; i ++ ) {
    for ( int j = 1; j < ynop - 1; j ++ ) {
      leftcell[i][j] = start_index;
      start_index ++;
    }
  }
  

  // ------ rightcell ----------
  // first row and last row of rightcell
  for ( int j = 0; j < ynop; j ++ ){
    rightcell[0][j] = topcell[j][xnop-1];
    rightcell[znop-1][j] = bottomcell[j][xnop-1];
  }

  // first colume and last colume of rightcell
  for ( int i = 0; i < znop; i ++ ) {
    rightcell[i][0] = backcell[i][xnop-1];
    rightcell[i][ynop-1] = frontcell[i][xnop-1];
  }
  
  for ( int i = 1; i < znop - 1; i ++ ) {
    for ( int j = 1; j < ynop - 1; j ++ ) {
      rightcell[i][j] = start_index;
      start_index ++;
    }
  }

  // output cell 
  
  // top surface
  for ( int i = 0; i < ynop-1; i ++ )
    for ( int j = 0; j < xnop-1; j ++ ) 
      out << " " << 4 << " " << topcell[i][j] << " " << topcell[i][j+1] <<
	" " << topcell[i+1][j+1] << " " << topcell[i+1][j] << endl;


  // bottom surface
  for ( int i = 0; i < ynop-1; i ++ )
    for ( int j = 0; j < xnop-1; j ++ ) 
      out << " " << 4 << " " << bottomcell[i][j] << " " << bottomcell[i][j+1] <<
	" " << bottomcell[i+1][j+1] << " " << bottomcell[i+1][j] << endl;

  
  // front surface
  for ( int i = 0; i < znop-1; i ++ )
    for ( int j = 0; j < xnop-1; j ++ ) 
      out << " " << 4 << " " << frontcell[i][j] << " " << frontcell[i][j+1] <<
	" " << frontcell[i+1][j+1] << " " << frontcell[i+1][j] << endl;


  // back surface
  for ( int i = 0; i < znop-1; i ++ )
    for ( int j = 0; j < xnop-1; j ++ ) 
      out << " " << 4 << " " << backcell[i][j] << " " << backcell[i][j+1] <<
	" " << backcell[i+1][j+1] << " " << backcell[i+1][j] << endl;

  
  // left surface
  for ( int i = 0; i < znop-1; i ++ )
    for ( int j = 0; j < ynop-1; j ++ ) 
      out << " " << 4 << " " << leftcell[i][j] << " " << leftcell[i][j+1] <<
	" " << leftcell[i+1][j+1] << " " << leftcell[i+1][j] << endl;


  // right surface
  for ( int i = 0; i < znop-1; i ++ )
    for ( int j = 0; j < ynop-1; j ++ ) 
      out << " " << 4 << " " << rightcell[i][j] << " " << rightcell[i][j+1] <<
	" " << rightcell[i+1][j+1] << " " << rightcell[i+1][j] << endl;

  out << endl;

  //----------------------------------------------------------------------
  //Celltypes
  out << "CELL_TYPES " << surfaceElementNo << endl;  
  for ( int i = 0; i < surfaceElementNo; i ++ )
    out << " " << 9 << endl; // structured
  
  out << endl;

  //---------------------------------------------------------------------
  //Celldatasets
  out << "CELL_DATA " << surfaceElementNo << endl;
  out << "FIELD DOMAIN 2" << endl;

  out << "SurfaceElement_HEAT_FLUX_q 1 " << surfaceElementNo << " FLOAT" << endl;
  // what happened with vector or scalar?
  
  for ( int i = 0; i < surfaceElementNo; i ++ )
    out << " " << q_surface[i] << endl;
  
  out << endl;
  
  out << "SurfaceElement_HEAT_Q 1 " << surfaceElementNo << " FLOAT " << endl;
  for ( int i = 0; i < surfaceElementNo; i ++ )
     out << " " << Q_surface[i] << endl;

  out.close();  
  
}


void MakeTableFunction::vtkVolTableMake(const char *_argv, const int &xnop,
					const int &ynop, const int &znop,
					const double *X, const double *Y,
					const double *Z,
					const int &VolElementNo,
					const double *qdiv,
					const double *Qdiv){


  ofstream out(_argv);// volume table ( div of heat flux and Q )
  
  //vtk format
  //-------------------------------------------------------------
  //Header
  out <<"# vtk DataFile Version 2.0" << endl;
  out <<"Three-D cube divergence of heat flux and Q in volume elements"<<endl;
  out <<"ASCII"<<endl;
  out << "DATASET UNSTRUCTURED_GRID"<<endl;
  out << endl;

  int pointNo;
  
  //----------------------------------------------------------------
  //PointIndex--xyz coordinates for each point
  pointNo = xnop * ynop * znop;
  out <<"POINTS  "<< pointNo << "  FLOAT" << endl;

  // top to bottom, left to right, back to front
  for ( int k = 0; k < znop; k ++ ) 
    for ( int j = 0; j < ynop; j ++ ) 
      for ( int i = 0; i < xnop ; i ++ ) 
	out << " " << X[i] << " " << Y[j] << " " << Z[k] << endl;

  out << endl;

  
  //-------------------------------------------------------------
  //CellIndex
  out << "CELLS " << VolElementNo << " " <<  VolElementNo * 9 << endl;
  double volcell[znop][ynop][xnop];
  int start_index = 0;
  
  for ( int k = 0; k < znop; k ++ )
    for ( int j = 0; j < ynop; j ++ )
      for ( int i = 0; i < xnop; i ++ ){
	volcell[k][j][i] = start_index;
	start_index ++;
      }

  for ( int k = 0; k < znop - 1; k ++ )
    for ( int j = 0; j < ynop -1 ; j ++ )
      for ( int i = 0; i < xnop - 1; i ++ )
	out << " " << 8 << " " <<
	  volcell[k][j][i] << " " << volcell[k][j][i+1] << " " <<
	  volcell[k][j+1][i+1] << " " << volcell[k][j+1][i] << " " <<
	  volcell[k+1][j][i] << " " << volcell[k+1][j][i+1] << " " <<
	  volcell[k+1][j+1][i+1] << " " << volcell[k+1][j+1][i] << endl;
  
  out << endl;
  
  //----------------------------------------------------------------------
  //Celltypes
  out << "CELL_TYPES " << VolElementNo << endl;
  for ( int i = 0; i < VolElementNo; i ++ )
    out << " " << 12 << endl;

  out << endl;
  
  
  //---------------------------------------------------------------------
  //Celldatasets
  out << "CELL_DATA " << VolElementNo << endl;
  out << "FIELD DOMAIN 2" << endl;

  //heat flux on surfaces; heat flux divergence in volumes; Q on surfaces and volumes

  out << "VOL_DIV_q 1 " << VolElementNo << " FLOAT " << endl;
  for ( int i = 0; i < VolElementNo; i ++ )
    out << " " << qdiv[i] << endl;

  out << endl;
  
  out << "Vol_HEAT_Q 1 " << VolElementNo << " FLOAT " << endl;
  for ( int i = 0; i < VolElementNo; i ++ )
    out << " " << Qdiv[i] << endl;
  
  out.close();
  
    
}

