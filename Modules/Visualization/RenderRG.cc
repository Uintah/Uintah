/*
 *  Render.cc: Routines that use the shear-warp VolPack library to
 *             volume render regularly gridded data.
 *
 *  Written by:
 *   Aleksandra Kuswik
 *   Department of Computer Science
 *   University of Utah
 *   May 1997
 *
 *  Copyright (C) 1997 SCI Group
 */

#define at() cerr << "@ " << __FILE__ << ":" << __LINE__ << endl

#include <iostream.h>
#include <Modules/Visualization/RenderRG.h>

RenderRGVolume::RenderRGVolume()
{
  int vpres;
  Voxel *dummy_voxel = new Voxel;

  
  /****************************************************************
   * Rendering Context
  ****************************************************************/

  // Create a new context
  vpc = vpCreateContext();
  if ( vpc == NULL ) cerr << "context is NULL\n";

  /****************************************************************
   * Voxel description
  ****************************************************************/

  // Declare the size of the voxel and the number of fields it contains
  vpres = vpSetVoxelSize ( vpc, sizeof(Voxel), VP__NUM_FIELDS,
			  VP__NUM_SHADE_FIELDS, VP__NUM_CLASSIFY_FIELDS );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelSize" << vpGetErrorString(vpGetError(vpc)) << "\n";

  // Declare the size and position of each field within the voxel
  // Normal
  vpres = vpSetVoxelField( vpc, VP__NORMAL_FIELD, sizeof(dummy_voxel->normal),
			  vpFieldOffset(dummy_voxel, normal), VP__NORMAL_MAX );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField1" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  // Gradient
  vpres = vpSetVoxelField(vpc, VP__GRAD_FIELD, sizeof(dummy_voxel->gradient),
			  vpFieldOffset(dummy_voxel, gradient), VP__GRAD_MAX );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField3" << vpGetErrorString(vpGetError(vpc)) << "\n";

  /****************************************************************
   * Additional settings
  ****************************************************************/
  
  // voxels of <=0.05 opacity are transparent
  vpres = vpSetd(vpc, VP_MIN_VOXEL_OPACITY, 0.05);
  if ( vpres != VP_OK )
    cerr << "vpSetd " << vpGetErrorString(vpGetError(vpc)) << endl;

  // set the threshhold to be 95%
  vpres = vpSetd(vpc, VP_MAX_RAY_OPACITY, 0.95);
  if ( vpres != VP_OK ) cerr << "vpSetd(Rendering) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

}

void
RenderRGVolume::NewScalarField( ScalarFieldRG *sfield, View& v )
{
  double smin, smax;
  int densitySize, volumeSize;
  int vpres;
  Voxel *dummy_voxel = new Voxel;

  // find the min and max scalar field values
  sfield->get_minmax(smin,smax);

  /****************************************************************
   * Volumes
  ****************************************************************/

  // Set volume dimensions
  vpres = vpSetVolumeSize( vpc, sfield->nx, sfield->ny, sfield->nz );
  if ( vpres != VP_OK )
    cerr << "vpSetVolumeSize" << vpGetErrorString(vpGetError(vpc)) << "\n";

  // Scalar
  vpres = vpSetVoxelField(vpc, VP__SCALAR_FIELD, sizeof(dummy_voxel->scalar),
			  vpFieldOffset(dummy_voxel, scalar), smax );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField2" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  // determine array sizes
  densitySize = sfield->nx * sfield->ny * sfield->nz;
  volumeSize = densitySize * sizeof(Voxel);
  
  // store the scalar values in a one dimensional array
  unsigned char *density = new unsigned char[densitySize];
  (sfield->grid).get_onedim_byte( density );

  // allocate memory for the volume
  Voxel * volume = new Voxel[densitySize];

  // check if memory was allocated
  if (density == NULL || volume == NULL) cerr << "out of memory\n";

  vpres = vpSetRawVoxels(vpc, volume, volumeSize, sizeof(Voxel),
			 sfield->nx * sizeof(Voxel),
			 sfield->nx * sfield->ny * sizeof(Voxel));
  if ( vpres != VP_OK ) cerr << "vpSetRawVoxels" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  vpres = vpVolumeNormals(vpc, density, densitySize, VP__SCALAR_FIELD,
			  VP__GRAD_FIELD, VP__NORMAL_FIELD);
  if ( vpres != VP_OK ) cerr << "vpVolumeNormals " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

}

void
RenderRGVolume::Process( ScalarFieldRG *sfield, ColorMapHandle cmap,
			View& v, int raster, unsigned char * a )
{
  int vpres;                      // error code
  Voxel *dummy_voxel = new Voxel; // voxel to initialize fields
  int volumeSize, densitySize, shadeTableSize; // various sizes

  double smin,smax;
  sfield->get_minmax(smin,smax);

#if 0
  /****************************************************************
   * Volumes
  ****************************************************************/

  // Set volume dimensions
  vpres = vpSetVolumeSize( vpc, sfield->nx, sfield->ny, sfield->nz );
  if ( vpres != VP_OK )
    cerr << "vpSetVolumeSize" << vpGetErrorString(vpGetError(vpc)) << "\n";

#if 0  
  // Declare the size of the voxel and the number of fields it contains
  vpres = vpSetVoxelSize ( vpc, sizeof(Voxel), VP__NUM_FIELDS,
			  VP__NUM_SHADE_FIELDS, VP__NUM_CLASSIFY_FIELDS );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelSize" << vpGetErrorString(vpGetError(vpc)) << "\n";

  // Declare the size and position of each field within the voxel
  // Normal
  vpres = vpSetVoxelField( vpc, VP__NORMAL_FIELD, sizeof(dummy_voxel->normal),
			  vpFieldOffset(dummy_voxel, normal), VP__NORMAL_MAX );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField1" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  // Scalar
  vpres = vpSetVoxelField(vpc, VP__SCALAR_FIELD, sizeof(dummy_voxel->scalar),
			  vpFieldOffset(dummy_voxel, scalar), smax );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField2" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  // Gradient
  vpres = vpSetVoxelField(vpc, VP__GRAD_FIELD, sizeof(dummy_voxel->gradient),
			  vpFieldOffset(dummy_voxel, gradient), VP__GRAD_MAX );
  if ( vpres != VP_OK )
    cerr << "vpSetVoxelField3" << vpGetErrorString(vpGetError(vpc)) << "\n";
#endif
  
  // determine array sizes
  densitySize = sfield->nx * sfield->ny * sfield->nz;
  volumeSize = densitySize * sizeof(Voxel);
  
  // store the scalar values in a one dimensional array
  unsigned char *density = new unsigned char[densitySize];
  (sfield->grid).get_onedim_byte( density );

  // allocate memory for the volume
  Voxel * volume = new Voxel[densitySize];

  // check if memory was allocated
  if (density == NULL || volume == NULL) cerr << "out of memory\n";

  vpres = vpSetRawVoxels(vpc, volume, volumeSize, sizeof(Voxel),
			 sfield->nx * sizeof(Voxel),
			 sfield->nx * sfield->ny * sizeof(Voxel));
  if ( vpres != VP_OK ) cerr << "vpSetRawVoxels" << vpGetErrorString(vpGetError(vpc)) << "\n";
  
  vpres = vpVolumeNormals(vpc, density, densitySize, VP__SCALAR_FIELD,
			  VP__GRAD_FIELD, VP__NORMAL_FIELD);
  if ( vpres != VP_OK ) cerr << "vpVolumeNormals " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
#endif  

  /****************************************************************
   * Classification
   ****************************************************************/

  // allocate the classification table
  float *scalarTable = new float[(int)smax+1];

  // find the difference in min and max scalar value
  double diff = smax - smin;
  
  Array1<int> xl;  // = *(cmap->rawRampAlphaT);
  Array1<float> sv = *(cmap->rawRampAlpha);

  // convert percentages to scalar values
  int j;
  for(j=0;j<sv.size();j++)
    xl.add( (int)( (*(cmap->rawRampAlphaT))[j] * diff) );

  //
  // make sure this works!
  //

  xl.remove_all();
  sv.remove_all();
  xl.add( 0 );
  xl.add( 24 );
  xl.add( 255 );
  sv.add( 0.0 );
  sv.add( 1.0 );
  sv.add( 1.0 );

  // place the classification table in the rendering context
  vpres = vpSetClassifierTable(vpc, 0, VP__SCALAR_FIELD, scalarTable,
			       (smax+1)*sizeof(float));
  if ( vpres != VP_OK )
    cerr <<"vpSetClassifierTable "<<vpGetErrorString(vpGetError(vpc)) << endl;

  // create the classification by specifying a ramp
  vpres = vpRamp( scalarTable, sizeof(float), xl.size(), xl.get_objs(),
		 sv.get_objs());
  if ( vpres != VP_OK )
    cerr <<"vpRamp "<<vpres <<" "<< vpGetErrorString(vpGetError(vpc)) << endl;

  //
  // additional gradient based classification
  //

  float gradTable[VP_GRAD_MAX+1];
  
  vpres = vpSetClassifierTable(vpc, 1, VP__GRAD_FIELD, gradTable,
			       (VP_GRAD_MAX+1)*sizeof(float));
  if ( vpres != VP_OK )
    cerr <<"vpSetClassifierTable "<<vpGetErrorString(vpGetError(vpc)) << endl;

  int GradientRampPoints = 4;
  int GradientRampX[] = {    0,   5,  20, 221};
  float GradientRampY[] = {0.0, 0.0, 1.0, 1.0};
  vpres = vpRamp(gradTable, sizeof(float), GradientRampPoints,
		 GradientRampX, GradientRampY);
  if ( vpres != VP_OK )
    cerr <<"vpRamp "<<vpres <<" "<< vpGetErrorString(vpGetError(vpc)) << endl;


  /****************************************************************
   * Classify volume
  ****************************************************************/

  vpres = vpClassifyVolume(vpc);
  if ( vpres != VP_OK ) cerr << "vpClassifyVolume " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  /****************************************************************
   * View Transformations
  ****************************************************************/

  // set pre-multiplication of matrices
  // OLA! not sure if it's supposed to be CONCAT_LEFT or right
  vpres = vpSeti(vpc, VP_CONCAT_MODE, VP_CONCAT_LEFT );
  if ( vpres != VP_OK ) cerr << "vpSeti " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  Point bboxmin, bboxmax;
  // find bbox min and max corners
  sfield->get_bounds( bboxmin, bboxmax );
  
  // set up the modeling matrix
  vpres = vpCurrentMatrix(vpc, VP_MODEL);
  if ( vpres != VP_OK ) cerr << "vpCurrentMatrix(1) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpIdentityMatrix(vpc);
  if ( vpres != VP_OK ) cerr << "vpIdentity(1) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpTranslate(vpc, bboxmin.x()-0.5, bboxmin.y()-0.5, bboxmin.z()-0.5);
  if ( vpres != VP_OK ) cerr << "vpTranslate(1b) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  // create the view transformation matrix
  Point eye;
  Vector z, up;
  eye = v.eyep();
  z = v.lookat() - eye;
  z.normalize();
  up = v.up() - z * Dot(z,v.up());
  up.normalize();
  Vector x = Cross(z,up);
  x.normalize();

  vpMatrix4 m;
  m[0][0] = x.x();
  m[0][1] = x.y();
  m[0][2] = x.z();
  m[0][3] = 0.;
  m[1][0] = -up.x();
  m[1][1] = -up.y();
  m[1][2] = -up.z();
  m[1][3] = 0.;
  m[2][0] = z.x();
  m[2][1] = z.y();
  m[2][2] = z.z();
  m[2][3] = 0.;
  m[3][0] = 0.;
  m[3][1] = 0.;
  m[3][2] = 0.;
  m[3][3] = 1.;

  // set up the view matrix
  vpres = vpCurrentMatrix(vpc, VP_VIEW);
  if ( vpres != VP_OK ) cerr << "vpCurrent(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpIdentityMatrix(vpc);
  if ( vpres != VP_OK ) cerr << "vpIdentity(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpTranslate(vpc, eye.x(), eye.y(), eye.z());
  if ( vpres != VP_OK ) cerr << "vpTranslate(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  vpres = vpMultMatrix(vpc, m);
  if ( vpres != VP_OK ) cerr << "vpMult(2) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  // set up the projection matrix
  vpres = vpCurrentMatrix(vpc, VP_PROJECT);
  if ( vpres != VP_OK ) cerr << "vpCurrent(3) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpWindow(vpc, VP_PARALLEL, -1, 1, -1, 1, -1, 1);
  if ( vpres != VP_OK ) cerr << "vpWindow(3) " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  /****************************************************************
   * Shading and lighting
  ****************************************************************/

  // allocate the shade table
  shadeTableSize = (VP_NORM_MAX+1)*VP__COLOR_CHANNELS*VP__NUM_MATERIALS;
  float *shadeTable = new float[shadeTableSize];

  // place the classification table in the rendering context
  vpres = vpSetLookupShader(vpc, VP__COLOR_CHANNELS, VP__NUM_MATERIALS,
			    VP__NORMAL_FIELD, shadeTable,
			    shadeTableSize*sizeof(float), 0, NULL, 0);
  if ( vpres != VP_OK )
    cerr << "SetLookup " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_AMBIENT, VP_BOTH_SIDES,
		0.18, 0.18, 0.18);
  if ( vpres != VP_OK )
    cerr << "SetMaterial(a) " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_DIFFUSE, VP_BOTH_SIDES,
		0.35, 0.35, 0.35);
  if ( vpres != VP_OK )
    cerr << "SetMaterial(b) " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_SPECULAR, VP_BOTH_SIDES,
		0.39, 0.39, 0.39);
  if ( vpres != VP_OK )
    cerr << "SetMaterial(c) " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetMaterial(vpc, VP_MATERIAL0, VP_SHINYNESS, VP_BOTH_SIDES,
			10.0,0.0,0.0);
  if ( vpres != VP_OK )
    cerr << "shine " << vpGetErrorString(vpGetError(vpc)) << endl;

//  Vector lookdir(v.lookat()-v.eyep());
  Vector lookdir(v.eyep()-v.lookat());
  vpres = vpSetLight(vpc, VP_LIGHT0, VP_DIRECTION,
		     lookdir.x(), lookdir.y(), lookdir.z());
  if ( vpres != VP_OK )
    cerr << "setlight " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpSetLight(vpc, VP_LIGHT0, VP_COLOR, 1.0, 1.0, 1.0);
  if ( vpres != VP_OK )
    cerr << "setlightcolor " << vpGetErrorString(vpGetError(vpc)) << endl;
  
  vpres = vpEnable(vpc, VP_LIGHT0, 1);
  if ( vpres != VP_OK )
    cerr << "enable " << vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpShadeTable(vpc);
  if ( vpres != VP_OK )
    cerr << "shadetable " << vpGetErrorString(vpGetError(vpc)) << endl;
  
#if 0  
  /****************************************************************
   * Depth cueing
  ****************************************************************/

    vpSetDepthCueing(vpc, 0.8, 0.8);
    vpEnable(vpc, VP_DEPTH_CUE, 1);
#endif  

  /****************************************************************
   * Images
  ****************************************************************/

  vpres = vpSetImage(vpc, a, raster, raster, raster * 3, VP_RGB);
  if ( vpres != VP_OK ) cerr << "vpSetImage " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  /****************************************************************
   * Rendering
  ****************************************************************/

  // render the volume
    vpres = vpRenderClassifiedVolume(vpc);
  if ( vpres != VP_OK ) cerr << "vpRenderClassified " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

#if 0  
  // the following is a bunch of zeros...
  int w, h;
  vpres = vpGeti( vpc, VP_INTERMEDIATE_WIDTH, &w );
  if ( vpres != VP_OK ) cerr << "AAA " <<
    vpGetErrorString(vpGetError(vpc)) << endl;

  vpres = vpGeti( vpc, VP_INTERMEDIATE_HEIGHT, &h );
  if ( vpres != VP_OK ) cerr << "BBB " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  
  unsigned char *joy = new unsigned char[w*h*3];
  vpres = vpGetImage(vpc, joy, w, h, w*3, VP_RGB, VP_IMAGE_BUFFER);
  if ( vpres != VP_OK ) cerr << "vpGetImage " <<
    vpGetErrorString(vpGetError(vpc)) << endl;
  #endif
}

