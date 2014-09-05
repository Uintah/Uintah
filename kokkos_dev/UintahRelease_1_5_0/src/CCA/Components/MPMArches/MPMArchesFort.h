/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//----- MPMArchesFort.h -----------------------------------------------

#ifndef Uintah_Component_Arches_MPMArchesFort_h
#define Uintah_Component_Arches_MPMArchesFort_h

/**************************************

HEADER
   MPMArchesFort
   
   Contains the header files to define interfaces between Fortran and
   C++ for Arches.

GENERAL INFORMATION

   MPMArchesFort.h

   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   
KEYWORDS
   Arches Fortran Interface

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

// GROUP: Function Definitions:
////////////////////////////////////////////////////////////////////////
//#define FORT_COLLECT_FCDRAG_TO_CC collect_drag_cc_
//#define FORT_MM_INTERP_CCTOFC interp_centertoface_
#define FORT_MM_INTERP_FCTOCC interp_facetocenter_
//#define FORT_MM_MOM_EXCH_CONT momentum_exchange_term_continuous_cc_
//#define FORT_MM_PRESSFORCE pressure_force_
#define FORT_MM_REDISTRIBUTE_DRAG redistribute_dragforce_cc_
// GROUP: Function Declarations:
////////////////////////////////////////////////////////////////////////

extern "C"
{

  //////////////////////////////////////////////////////////////////////
  // Collect drag sources for the gas phase from faces to cell center

#if 0
  void
  FORT_COLLECT_FCDRAG_TO_CC(
			    double* su_dragx_cc, 
			    double* sp_dragx_cc,
   			    double* su_dragx_fcy, 
			    double* sp_dragx_fcy,
			    const int* ioff, 
			    const int* joff, 
			    const int*koff,
			    const int* dim_lo_su_cc,
			    const int* dim_hi_su_cc,
			    const int* dim_lo_sp_cc,
			    const int* dim_hi_sp_cc,
			    const int* dim_lo_su_fcy,
			    const int* dim_hi_su_fcy,
			    const int* dim_lo_sp_fcy,
			    const int* dim_hi_sp_fcy,
			    const int* valid_lo,
			    const int* valid_hi);
#endif


  //////////////////////////////////////////////////////////////////////
  // Interpolate a quantity from the cell center to the face center

#if 0
  void 
  FORT_MM_INTERP_CCTOFC(
			double* phi_fc,
			double* phi_cc,
			const int* ioff, const int* joff, const int* koff,
			const int* dim_lo_fc, const int* dim_hi_fc,
			const int* dim_lo_cc, const int* dim_hi_cc,
			const int* valid_lo, const int* valid_hi);
#endif

  //////////////////////////////////////////////////////////////////////
  // Interpolate a quantity from the face center to the cell center

  void
  FORT_MM_INTERP_FCTOCC(
			double* phi_cc,
			double* phi_fc,
			double* efac, double* wfac,
			const int* ioff, const int* joff, const int* koff, 
			const int* index,
			const int* dim_lo, const int* dim_hi,
			const int* dim_lo_cc, const int* dim_hi_cc,
			const int* dim_lo_fc, const int* dim_hi_fc,
			const int* valid_lo, const int* valid_hi);
			
#if 0
  //////////////////////////////////////////////////////////////////////
  // Calculate momentum exchange terms for continuous solid-fluid
  // interaction

  void 
  FORT_MM_MOM_EXCH_CONT(
			double* uVelNonlinearSrc_fcy,
			double* uVelLinearSrc_fcy,
			double* uVelNonlinearSrc_fcz,
			double* uVelLinearSrc_fcz,
			double* uVelNonlinearSrc_cc,
			double* uVelLinearSrc_cc,
			double* dragforcex_fcy,
			double* dragforcex_fcz,
			double* dragforcex_cc,
			double* ug_cc,
			double* up_cc,
			double* up_fcy,
			double* up_fcz,
			double* epsg,
			double* epsg_solid,
			double* viscos,
			double* csmag,
			double* sew, double* sns, double* stb,
			double* yy, double* zz, double* yv, double* zw,
			const int* dim_lo, const int* dim_hi,
			const int* dim_lo_su_fcy, const int* dim_hi_su_fcy,
			const int* dim_lo_sp_fcy, const int* dim_hi_sp_fcy,
			const int* dim_lo_su_fcz, const int* dim_hi_su_fcz,
			const int* dim_lo_sp_fcz, const int* dim_hi_sp_fcz,
			const int* dim_lo_su_cc, const int* dim_hi_su_cc,
			const int* dim_lo_sp_cc, const int* dim_hi_sp_cc,
			const int* dim_lo_dx_fcy, const int* dim_hi_dx_fcy,
			const int* dim_lo_dx_fcz, const int* dim_hi_dx_fcz,
			const int* dim_lo_dx_cc, const int* dim_hi_dx_cc,
			const int* dim_lo_ugc, const int* dim_hi_ugc,
			const int* dim_lo_upc, const int* dim_hi_upc,
			const int* dim_lo_upy, const int* dim_hi_upy,
			const int* dim_lo_upz, const int* dim_hi_upz,
			const int* dim_lo_eps, const int* dim_hi_eps,
			const int* dim_lo_epss, const int* dim_hi_epss,
			const int* valid_lo, const int* valid_hi,
			const int* ioff, const int* joff, const int* koff,
			const int* indexflo,
			const int* indext1,
			const int* indext2,
			const int* pcell, 
			const int* mmwallid, 
			const int*ffield);
#endif

  //////////////////////////////////////////////////////////////////////
  // Calculate pressure forces on continuous solid from gas

#if 0
  void
  FORT_MM_PRESSFORCE(
		     double* pressforcex_fcx,
		     double* pressforcey_fcy,
		     double* pressforcez_fcz,
		     double* epsg,
		     double* epsg_solid,
		     double* pres,
		     double* sew, double*sns, double*stb,
		     const int* dim_lo, const int* dim_hi,
		     const int* dim_lo_fcx, const int* dim_hi_fcx,
		     const int* dim_lo_fcy, const int* dim_hi_fcy,
		     const int* dim_lo_fcz, const int* dim_hi_fcz,
		     const int* dim_lo_eps, const int* dim_hi_eps,
		     const int* dim_lo_epss, const int* dim_hi_epss,
		     const int* dim_lo_p, const int* dim_hi_p,
		     const int* valid_lo, const int* valid_hi,
		     const int* pcell, const int* wall, const int* ffield);
#endif

  //////////////////////////////////////////////////////////////////////
  // Redistribute drag forces on continuous solid calculated at
  // cell centers (for partially filled cells) to face centers

  void
  FORT_MM_REDISTRIBUTE_DRAG(
			    double* dragforce_fcx,
			    double* dragforce_cc,
			    const int* ioff, const int* joff, const int* koff,
			    const int* dim_lo_fcx, const int* dim_hi_fcx,
			    const int* dim_lo_cc, const int* dim_hi_cc,
			    const int* valid_lo, const int* valid_hi);

}

#endif

