
#ifndef fspec_read_complex_geometry
#define fspec_read_complex_geometry

#ifdef __cplusplus

extern "C" void read_complex_geometry_(int* iccst,
                                       int* jccst,
                                       int* kccst,
                                       int* inext_low_x, int* inext_low_y, int* inext_low_z, int* inext_high_x, int* inext_high_y, int* inext_high_z, int* inext_ptr,
                                       int* jnext_low_x, int* jnext_low_y, int* jnext_low_z, int* jnext_high_x, int* jnext_high_y, int* jnext_high_z, int* jnext_ptr,
                                       int* knext_low_x, int* knext_low_y, int* knext_low_z, int* knext_high_x, int* knext_high_y, int* knext_high_z, int* knext_ptr,
                                       int* epsg_low_x, int* epsg_low_y, int* epsg_low_z, int* epsg_high_x, int* epsg_high_y, int* epsg_high_z, double* epsg_ptr,
                                       int* totarea_low_x, int* totarea_low_y, int* totarea_low_z, int* totarea_high_x, int* totarea_high_y, int* totarea_high_z, double* totarea_ptr,
                                       int* nbar1_low_x, int* nbar1_low_y, int* nbar1_low_z, int* nbar1_high_x, int* nbar1_high_y, int* nbar1_high_z, double* nbar1_ptr,
                                       int* nbar2_low_x, int* nbar2_low_y, int* nbar2_low_z, int* nbar2_high_x, int* nbar2_high_y, int* nbar2_high_z, double* nbar2_ptr,
                                       int* nbar3_low_x, int* nbar3_low_y, int* nbar3_low_z, int* nbar3_high_x, int* nbar3_high_y, int* nbar3_high_z, double* nbar3_ptr,
                                       int* cbar1_low_x, int* cbar1_low_y, int* cbar1_low_z, int* cbar1_high_x, int* cbar1_high_y, int* cbar1_high_z, double* cbar1_ptr,
                                       int* cbar2_low_x, int* cbar2_low_y, int* cbar2_low_z, int* cbar2_high_x, int* cbar2_high_y, int* cbar2_high_z, double* cbar2_ptr,
                                       int* cbar3_low_x, int* cbar3_low_y, int* cbar3_low_z, int* cbar3_high_x, int* cbar3_high_y, int* cbar3_high_z, double* cbar3_ptr,
                                       int* gaf_x_low_x, int* gaf_x_low_y, int* gaf_x_low_z, int* gaf_x_high_x, int* gaf_x_high_y, int* gaf_x_high_z, double* gaf_x_ptr,
                                       int* gaf_xe_low_x, int* gaf_xe_low_y, int* gaf_xe_low_z, int* gaf_xe_high_x, int* gaf_xe_high_y, int* gaf_xe_high_z, double* gaf_xe_ptr,
                                       int* gaf_y_low_x, int* gaf_y_low_y, int* gaf_y_low_z, int* gaf_y_high_x, int* gaf_y_high_y, int* gaf_y_high_z, double* gaf_y_ptr,
                                       int* gaf_yn_low_x, int* gaf_yn_low_y, int* gaf_yn_low_z, int* gaf_yn_high_x, int* gaf_yn_high_y, int* gaf_yn_high_z, double* gaf_yn_ptr,
                                       int* gaf_z_low_x, int* gaf_z_low_y, int* gaf_z_low_z, int* gaf_z_high_x, int* gaf_z_high_y, int* gaf_z_high_z, double* gaf_z_ptr,
                                       int* gaf_zt_low_x, int* gaf_zt_low_y, int* gaf_zt_low_z, int* gaf_zt_high_x, int* gaf_zt_high_y, int* gaf_zt_high_z, double* gaf_zt_ptr,
                                       int* patchindex,
                                       int* tot_cutp);

static void fort_read_complex_geometry( int & iccst,
                                        int & jccst,
                                        int & kccst,
                                        Uintah::CCVariable<int> & inext,
                                        Uintah::CCVariable<int> & jnext,
                                        Uintah::CCVariable<int> & knext,
                                        Uintah::CCVariable<double> & epsg,
                                        Uintah::CCVariable<double> & totarea,
                                        Uintah::CCVariable<double> & nbar1,
                                        Uintah::CCVariable<double> & nbar2,
                                        Uintah::CCVariable<double> & nbar3,
                                        Uintah::CCVariable<double> & cbar1,
                                        Uintah::CCVariable<double> & cbar2,
                                        Uintah::CCVariable<double> & cbar3,
                                        Uintah::CCVariable<double> & gaf_x,
                                        Uintah::CCVariable<double> & gaf_xe,
                                        Uintah::CCVariable<double> & gaf_y,
                                        Uintah::CCVariable<double> & gaf_yn,
                                        Uintah::CCVariable<double> & gaf_z,
                                        Uintah::CCVariable<double> & gaf_zt,
                                        int & patchindex,
                                        int & tot_cutp )
{
  Uintah::IntVector inext_low = inext.getWindow()->getOffset();
  Uintah::IntVector inext_high = inext.getWindow()->getData()->size() + inext_low - Uintah::IntVector(1, 1, 1);
  int inext_low_x = inext_low.x();
  int inext_high_x = inext_high.x();
  int inext_low_y = inext_low.y();
  int inext_high_y = inext_high.y();
  int inext_low_z = inext_low.z();
  int inext_high_z = inext_high.z();
  Uintah::IntVector jnext_low = jnext.getWindow()->getOffset();
  Uintah::IntVector jnext_high = jnext.getWindow()->getData()->size() + jnext_low - Uintah::IntVector(1, 1, 1);
  int jnext_low_x = jnext_low.x();
  int jnext_high_x = jnext_high.x();
  int jnext_low_y = jnext_low.y();
  int jnext_high_y = jnext_high.y();
  int jnext_low_z = jnext_low.z();
  int jnext_high_z = jnext_high.z();
  Uintah::IntVector knext_low = knext.getWindow()->getOffset();
  Uintah::IntVector knext_high = knext.getWindow()->getData()->size() + knext_low - Uintah::IntVector(1, 1, 1);
  int knext_low_x = knext_low.x();
  int knext_high_x = knext_high.x();
  int knext_low_y = knext_low.y();
  int knext_high_y = knext_high.y();
  int knext_low_z = knext_low.z();
  int knext_high_z = knext_high.z();
  Uintah::IntVector epsg_low = epsg.getWindow()->getOffset();
  Uintah::IntVector epsg_high = epsg.getWindow()->getData()->size() + epsg_low - Uintah::IntVector(1, 1, 1);
  int epsg_low_x = epsg_low.x();
  int epsg_high_x = epsg_high.x();
  int epsg_low_y = epsg_low.y();
  int epsg_high_y = epsg_high.y();
  int epsg_low_z = epsg_low.z();
  int epsg_high_z = epsg_high.z();
  Uintah::IntVector totarea_low = totarea.getWindow()->getOffset();
  Uintah::IntVector totarea_high = totarea.getWindow()->getData()->size() + totarea_low - Uintah::IntVector(1, 1, 1);
  int totarea_low_x = totarea_low.x();
  int totarea_high_x = totarea_high.x();
  int totarea_low_y = totarea_low.y();
  int totarea_high_y = totarea_high.y();
  int totarea_low_z = totarea_low.z();
  int totarea_high_z = totarea_high.z();
  Uintah::IntVector nbar1_low = nbar1.getWindow()->getOffset();
  Uintah::IntVector nbar1_high = nbar1.getWindow()->getData()->size() + nbar1_low - Uintah::IntVector(1, 1, 1);
  int nbar1_low_x = nbar1_low.x();
  int nbar1_high_x = nbar1_high.x();
  int nbar1_low_y = nbar1_low.y();
  int nbar1_high_y = nbar1_high.y();
  int nbar1_low_z = nbar1_low.z();
  int nbar1_high_z = nbar1_high.z();
  Uintah::IntVector nbar2_low = nbar2.getWindow()->getOffset();
  Uintah::IntVector nbar2_high = nbar2.getWindow()->getData()->size() + nbar2_low - Uintah::IntVector(1, 1, 1);
  int nbar2_low_x = nbar2_low.x();
  int nbar2_high_x = nbar2_high.x();
  int nbar2_low_y = nbar2_low.y();
  int nbar2_high_y = nbar2_high.y();
  int nbar2_low_z = nbar2_low.z();
  int nbar2_high_z = nbar2_high.z();
  Uintah::IntVector nbar3_low = nbar3.getWindow()->getOffset();
  Uintah::IntVector nbar3_high = nbar3.getWindow()->getData()->size() + nbar3_low - Uintah::IntVector(1, 1, 1);
  int nbar3_low_x = nbar3_low.x();
  int nbar3_high_x = nbar3_high.x();
  int nbar3_low_y = nbar3_low.y();
  int nbar3_high_y = nbar3_high.y();
  int nbar3_low_z = nbar3_low.z();
  int nbar3_high_z = nbar3_high.z();
  Uintah::IntVector cbar1_low = cbar1.getWindow()->getOffset();
  Uintah::IntVector cbar1_high = cbar1.getWindow()->getData()->size() + cbar1_low - Uintah::IntVector(1, 1, 1);
  int cbar1_low_x = cbar1_low.x();
  int cbar1_high_x = cbar1_high.x();
  int cbar1_low_y = cbar1_low.y();
  int cbar1_high_y = cbar1_high.y();
  int cbar1_low_z = cbar1_low.z();
  int cbar1_high_z = cbar1_high.z();
  Uintah::IntVector cbar2_low = cbar2.getWindow()->getOffset();
  Uintah::IntVector cbar2_high = cbar2.getWindow()->getData()->size() + cbar2_low - Uintah::IntVector(1, 1, 1);
  int cbar2_low_x = cbar2_low.x();
  int cbar2_high_x = cbar2_high.x();
  int cbar2_low_y = cbar2_low.y();
  int cbar2_high_y = cbar2_high.y();
  int cbar2_low_z = cbar2_low.z();
  int cbar2_high_z = cbar2_high.z();
  Uintah::IntVector cbar3_low = cbar3.getWindow()->getOffset();
  Uintah::IntVector cbar3_high = cbar3.getWindow()->getData()->size() + cbar3_low - Uintah::IntVector(1, 1, 1);
  int cbar3_low_x = cbar3_low.x();
  int cbar3_high_x = cbar3_high.x();
  int cbar3_low_y = cbar3_low.y();
  int cbar3_high_y = cbar3_high.y();
  int cbar3_low_z = cbar3_low.z();
  int cbar3_high_z = cbar3_high.z();
  Uintah::IntVector gaf_x_low = gaf_x.getWindow()->getOffset();
  Uintah::IntVector gaf_x_high = gaf_x.getWindow()->getData()->size() + gaf_x_low - Uintah::IntVector(1, 1, 1);
  int gaf_x_low_x = gaf_x_low.x();
  int gaf_x_high_x = gaf_x_high.x();
  int gaf_x_low_y = gaf_x_low.y();
  int gaf_x_high_y = gaf_x_high.y();
  int gaf_x_low_z = gaf_x_low.z();
  int gaf_x_high_z = gaf_x_high.z();
  Uintah::IntVector gaf_xe_low = gaf_xe.getWindow()->getOffset();
  Uintah::IntVector gaf_xe_high = gaf_xe.getWindow()->getData()->size() + gaf_xe_low - Uintah::IntVector(1, 1, 1);
  int gaf_xe_low_x = gaf_xe_low.x();
  int gaf_xe_high_x = gaf_xe_high.x();
  int gaf_xe_low_y = gaf_xe_low.y();
  int gaf_xe_high_y = gaf_xe_high.y();
  int gaf_xe_low_z = gaf_xe_low.z();
  int gaf_xe_high_z = gaf_xe_high.z();
  Uintah::IntVector gaf_y_low = gaf_y.getWindow()->getOffset();
  Uintah::IntVector gaf_y_high = gaf_y.getWindow()->getData()->size() + gaf_y_low - Uintah::IntVector(1, 1, 1);
  int gaf_y_low_x = gaf_y_low.x();
  int gaf_y_high_x = gaf_y_high.x();
  int gaf_y_low_y = gaf_y_low.y();
  int gaf_y_high_y = gaf_y_high.y();
  int gaf_y_low_z = gaf_y_low.z();
  int gaf_y_high_z = gaf_y_high.z();
  Uintah::IntVector gaf_yn_low = gaf_yn.getWindow()->getOffset();
  Uintah::IntVector gaf_yn_high = gaf_yn.getWindow()->getData()->size() + gaf_yn_low - Uintah::IntVector(1, 1, 1);
  int gaf_yn_low_x = gaf_yn_low.x();
  int gaf_yn_high_x = gaf_yn_high.x();
  int gaf_yn_low_y = gaf_yn_low.y();
  int gaf_yn_high_y = gaf_yn_high.y();
  int gaf_yn_low_z = gaf_yn_low.z();
  int gaf_yn_high_z = gaf_yn_high.z();
  Uintah::IntVector gaf_z_low = gaf_z.getWindow()->getOffset();
  Uintah::IntVector gaf_z_high = gaf_z.getWindow()->getData()->size() + gaf_z_low - Uintah::IntVector(1, 1, 1);
  int gaf_z_low_x = gaf_z_low.x();
  int gaf_z_high_x = gaf_z_high.x();
  int gaf_z_low_y = gaf_z_low.y();
  int gaf_z_high_y = gaf_z_high.y();
  int gaf_z_low_z = gaf_z_low.z();
  int gaf_z_high_z = gaf_z_high.z();
  Uintah::IntVector gaf_zt_low = gaf_zt.getWindow()->getOffset();
  Uintah::IntVector gaf_zt_high = gaf_zt.getWindow()->getData()->size() + gaf_zt_low - Uintah::IntVector(1, 1, 1);
  int gaf_zt_low_x = gaf_zt_low.x();
  int gaf_zt_high_x = gaf_zt_high.x();
  int gaf_zt_low_y = gaf_zt_low.y();
  int gaf_zt_high_y = gaf_zt_high.y();
  int gaf_zt_low_z = gaf_zt_low.z();
  int gaf_zt_high_z = gaf_zt_high.z();
  read_complex_geometry_( &iccst,
                          &jccst,
                          &kccst,
                          &inext_low_x, &inext_low_y, &inext_low_z, &inext_high_x, &inext_high_y, &inext_high_z, inext.getPointer(),
                          &jnext_low_x, &jnext_low_y, &jnext_low_z, &jnext_high_x, &jnext_high_y, &jnext_high_z, jnext.getPointer(),
                          &knext_low_x, &knext_low_y, &knext_low_z, &knext_high_x, &knext_high_y, &knext_high_z, knext.getPointer(),
                          &epsg_low_x, &epsg_low_y, &epsg_low_z, &epsg_high_x, &epsg_high_y, &epsg_high_z, epsg.getPointer(),
                          &totarea_low_x, &totarea_low_y, &totarea_low_z, &totarea_high_x, &totarea_high_y, &totarea_high_z, totarea.getPointer(),
                          &nbar1_low_x, &nbar1_low_y, &nbar1_low_z, &nbar1_high_x, &nbar1_high_y, &nbar1_high_z, nbar1.getPointer(),
                          &nbar2_low_x, &nbar2_low_y, &nbar2_low_z, &nbar2_high_x, &nbar2_high_y, &nbar2_high_z, nbar2.getPointer(),
                          &nbar3_low_x, &nbar3_low_y, &nbar3_low_z, &nbar3_high_x, &nbar3_high_y, &nbar3_high_z, nbar3.getPointer(),
                          &cbar1_low_x, &cbar1_low_y, &cbar1_low_z, &cbar1_high_x, &cbar1_high_y, &cbar1_high_z, cbar1.getPointer(),
                          &cbar2_low_x, &cbar2_low_y, &cbar2_low_z, &cbar2_high_x, &cbar2_high_y, &cbar2_high_z, cbar2.getPointer(),
                          &cbar3_low_x, &cbar3_low_y, &cbar3_low_z, &cbar3_high_x, &cbar3_high_y, &cbar3_high_z, cbar3.getPointer(),
                          &gaf_x_low_x, &gaf_x_low_y, &gaf_x_low_z, &gaf_x_high_x, &gaf_x_high_y, &gaf_x_high_z, gaf_x.getPointer(),
                          &gaf_xe_low_x, &gaf_xe_low_y, &gaf_xe_low_z, &gaf_xe_high_x, &gaf_xe_high_y, &gaf_xe_high_z, gaf_xe.getPointer(),
                          &gaf_y_low_x, &gaf_y_low_y, &gaf_y_low_z, &gaf_y_high_x, &gaf_y_high_y, &gaf_y_high_z, gaf_y.getPointer(),
                          &gaf_yn_low_x, &gaf_yn_low_y, &gaf_yn_low_z, &gaf_yn_high_x, &gaf_yn_high_y, &gaf_yn_high_z, gaf_yn.getPointer(),
                          &gaf_z_low_x, &gaf_z_low_y, &gaf_z_low_z, &gaf_z_high_x, &gaf_z_high_y, &gaf_z_high_z, gaf_z.getPointer(),
                          &gaf_zt_low_x, &gaf_zt_low_y, &gaf_zt_low_z, &gaf_zt_high_x, &gaf_zt_high_y, &gaf_zt_high_z, gaf_zt.getPointer(),
                          &patchindex,
                          &tot_cutp );
}

#else /* !__cplusplus */

C This is the FORTRAN code portion of the file:

      subroutine read_complex_geometry(iccst, jccst, kccst, inext_low_x
     & , inext_low_y, inext_low_z, inext_high_x, inext_high_y, 
     & inext_high_z, inext, jnext_low_x, jnext_low_y, jnext_low_z, 
     & jnext_high_x, jnext_high_y, jnext_high_z, jnext, knext_low_x, 
     & knext_low_y, knext_low_z, knext_high_x, knext_high_y, 
     & knext_high_z, knext, epsg_low_x, epsg_low_y, epsg_low_z, 
     & epsg_high_x, epsg_high_y, epsg_high_z, epsg, totarea_low_x, 
     & totarea_low_y, totarea_low_z, totarea_high_x, totarea_high_y, 
     & totarea_high_z, totarea, nbar1_low_x, nbar1_low_y, nbar1_low_z, 
     & nbar1_high_x, nbar1_high_y, nbar1_high_z, nbar1, nbar2_low_x, 
     & nbar2_low_y, nbar2_low_z, nbar2_high_x, nbar2_high_y, 
     & nbar2_high_z, nbar2, nbar3_low_x, nbar3_low_y, nbar3_low_z, 
     & nbar3_high_x, nbar3_high_y, nbar3_high_z, nbar3, cbar1_low_x, 
     & cbar1_low_y, cbar1_low_z, cbar1_high_x, cbar1_high_y, 
     & cbar1_high_z, cbar1, cbar2_low_x, cbar2_low_y, cbar2_low_z, 
     & cbar2_high_x, cbar2_high_y, cbar2_high_z, cbar2, cbar3_low_x, 
     & cbar3_low_y, cbar3_low_z, cbar3_high_x, cbar3_high_y, 
     & cbar3_high_z, cbar3, gaf_x_low_x, gaf_x_low_y, gaf_x_low_z, 
     & gaf_x_high_x, gaf_x_high_y, gaf_x_high_z, gaf_x, gaf_xe_low_x, 
     & gaf_xe_low_y, gaf_xe_low_z, gaf_xe_high_x, gaf_xe_high_y, 
     & gaf_xe_high_z, gaf_xe, gaf_y_low_x, gaf_y_low_y, gaf_y_low_z, 
     & gaf_y_high_x, gaf_y_high_y, gaf_y_high_z, gaf_y, gaf_yn_low_x, 
     & gaf_yn_low_y, gaf_yn_low_z, gaf_yn_high_x, gaf_yn_high_y, 
     & gaf_yn_high_z, gaf_yn, gaf_z_low_x, gaf_z_low_y, gaf_z_low_z, 
     & gaf_z_high_x, gaf_z_high_y, gaf_z_high_z, gaf_z, gaf_zt_low_x, 
     & gaf_zt_low_y, gaf_zt_low_z, gaf_zt_high_x, gaf_zt_high_y, 
     & gaf_zt_high_z, gaf_zt, patchindex, tot_cutp)

      implicit none
      integer iccst
      integer jccst
      integer kccst
      integer inext_low_x, inext_low_y, inext_low_z, inext_high_x, 
     & inext_high_y, inext_high_z
      integer inext(inext_low_x:inext_high_x, inext_low_y:inext_high_y,
     &  inext_low_z:inext_high_z)
      integer jnext_low_x, jnext_low_y, jnext_low_z, jnext_high_x, 
     & jnext_high_y, jnext_high_z
      integer jnext(jnext_low_x:jnext_high_x, jnext_low_y:jnext_high_y,
     &  jnext_low_z:jnext_high_z)
      integer knext_low_x, knext_low_y, knext_low_z, knext_high_x, 
     & knext_high_y, knext_high_z
      integer knext(knext_low_x:knext_high_x, knext_low_y:knext_high_y,
     &  knext_low_z:knext_high_z)
      integer epsg_low_x, epsg_low_y, epsg_low_z, epsg_high_x, 
     & epsg_high_y, epsg_high_z
      double precision epsg(epsg_low_x:epsg_high_x, epsg_low_y:
     & epsg_high_y, epsg_low_z:epsg_high_z)
      integer totarea_low_x, totarea_low_y, totarea_low_z, 
     & totarea_high_x, totarea_high_y, totarea_high_z
      double precision totarea(totarea_low_x:totarea_high_x, 
     & totarea_low_y:totarea_high_y, totarea_low_z:totarea_high_z)
      integer nbar1_low_x, nbar1_low_y, nbar1_low_z, nbar1_high_x, 
     & nbar1_high_y, nbar1_high_z
      double precision nbar1(nbar1_low_x:nbar1_high_x, nbar1_low_y:
     & nbar1_high_y, nbar1_low_z:nbar1_high_z)
      integer nbar2_low_x, nbar2_low_y, nbar2_low_z, nbar2_high_x, 
     & nbar2_high_y, nbar2_high_z
      double precision nbar2(nbar2_low_x:nbar2_high_x, nbar2_low_y:
     & nbar2_high_y, nbar2_low_z:nbar2_high_z)
      integer nbar3_low_x, nbar3_low_y, nbar3_low_z, nbar3_high_x, 
     & nbar3_high_y, nbar3_high_z
      double precision nbar3(nbar3_low_x:nbar3_high_x, nbar3_low_y:
     & nbar3_high_y, nbar3_low_z:nbar3_high_z)
      integer cbar1_low_x, cbar1_low_y, cbar1_low_z, cbar1_high_x, 
     & cbar1_high_y, cbar1_high_z
      double precision cbar1(cbar1_low_x:cbar1_high_x, cbar1_low_y:
     & cbar1_high_y, cbar1_low_z:cbar1_high_z)
      integer cbar2_low_x, cbar2_low_y, cbar2_low_z, cbar2_high_x, 
     & cbar2_high_y, cbar2_high_z
      double precision cbar2(cbar2_low_x:cbar2_high_x, cbar2_low_y:
     & cbar2_high_y, cbar2_low_z:cbar2_high_z)
      integer cbar3_low_x, cbar3_low_y, cbar3_low_z, cbar3_high_x, 
     & cbar3_high_y, cbar3_high_z
      double precision cbar3(cbar3_low_x:cbar3_high_x, cbar3_low_y:
     & cbar3_high_y, cbar3_low_z:cbar3_high_z)
      integer gaf_x_low_x, gaf_x_low_y, gaf_x_low_z, gaf_x_high_x, 
     & gaf_x_high_y, gaf_x_high_z
      double precision gaf_x(gaf_x_low_x:gaf_x_high_x, gaf_x_low_y:
     & gaf_x_high_y, gaf_x_low_z:gaf_x_high_z)
      integer gaf_xe_low_x, gaf_xe_low_y, gaf_xe_low_z, gaf_xe_high_x, 
     & gaf_xe_high_y, gaf_xe_high_z
      double precision gaf_xe(gaf_xe_low_x:gaf_xe_high_x, gaf_xe_low_y:
     & gaf_xe_high_y, gaf_xe_low_z:gaf_xe_high_z)
      integer gaf_y_low_x, gaf_y_low_y, gaf_y_low_z, gaf_y_high_x, 
     & gaf_y_high_y, gaf_y_high_z
      double precision gaf_y(gaf_y_low_x:gaf_y_high_x, gaf_y_low_y:
     & gaf_y_high_y, gaf_y_low_z:gaf_y_high_z)
      integer gaf_yn_low_x, gaf_yn_low_y, gaf_yn_low_z, gaf_yn_high_x, 
     & gaf_yn_high_y, gaf_yn_high_z
      double precision gaf_yn(gaf_yn_low_x:gaf_yn_high_x, gaf_yn_low_y:
     & gaf_yn_high_y, gaf_yn_low_z:gaf_yn_high_z)
      integer gaf_z_low_x, gaf_z_low_y, gaf_z_low_z, gaf_z_high_x, 
     & gaf_z_high_y, gaf_z_high_z
      double precision gaf_z(gaf_z_low_x:gaf_z_high_x, gaf_z_low_y:
     & gaf_z_high_y, gaf_z_low_z:gaf_z_high_z)
      integer gaf_zt_low_x, gaf_zt_low_y, gaf_zt_low_z, gaf_zt_high_x, 
     & gaf_zt_high_y, gaf_zt_high_z
      double precision gaf_zt(gaf_zt_low_x:gaf_zt_high_x, gaf_zt_low_y:
     & gaf_zt_high_y, gaf_zt_low_z:gaf_zt_high_z)
      integer patchindex
      integer tot_cutp
#endif /* __cplusplus */

#endif /* fspec_read_complex_geometry */

#ifndef PASS1
#  define PASS1(x) x/**/_low, x/**/_high, x
#endif

#ifndef PASS3
#  define PASS3A(x) x/**/_low_x, x/**/_low_y, x/**/_low_z, 
#  define PASS3B(x) x/**/_high_x, x/**/_high_y, x/**/_high_z, x
#endif
