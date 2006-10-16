/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


/*
 *  HeuristicStreamLines.h:  Heuristic approach for streamline creation of streamlines
 *
 *  Written by:
 *   Elisha Hughes, Frank B. Sachse
 *   CVRTI
 *   University of Utah
 *   Feb 2004, Jul 2004, Jul 2005
 */

#if !defined(HeuristicStreamLines_h)
#define HeuristicStreamLines_h

#include <Dataflow/Network/Module.h>
#include <Core/Thread/Thread.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/Field.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/CrvLinearLgn.h>
#include <Core/Datatypes/CurveMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/SampleField.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

typedef PointCloudMesh<ConstantBasis<Point> > HSLPoints_Mesh;
typedef HSLPoints_Mesh::Node HSLPoints_LPTS;
typedef GenericField<HSLPoints_Mesh, ConstantBasis<double>, vector<double> > HSLPoints_FPTS;

typedef CurveMesh<CrvLinearLgn<Point> > HSLStreamLines_Mesh;
typedef HSLStreamLines_Mesh::Node HSLStreamLines_LSL;
typedef HSLStreamLines_Mesh::handle_type HSLStreamLines_SLHandle;
typedef GenericField<HSLStreamLines_Mesh, CrvLinearLgn<double>, vector<double> > HSLStreamLines_FSL;

class HeuristicStreamLinesData
{
public:
  HSLStreamLines_FSL *msl;
  HSLStreamLines_FSL *cf;

  FieldHandle seed_fieldH;
  FieldHandle compare_fieldH;
  ScalarFieldInterfaceHandle cfi;

  FieldHandle src_fieldH;
  ScalarFieldInterfaceHandle sfi;
  FieldHandle weighting_fieldH;
  ScalarFieldInterfaceHandle wfi;
  FieldHandle pts_fieldH;
  VectorFieldInterfaceHandle vfi;

  int numsl; // number of streamlines
  int numpts;  // number of trials
  double minper; // mininal rendering radius in percent of max. bounding box length
  double maxper; // maximal rendering radius in percent of max. bounding box length
  double ming; // minimum for clamping and normalizing of field magnitude
  double maxg;  // maximum for clamping and normalizing of field magnitude
  double maxlen; // max. bounding box length
  int numsamples; // number of samples for 1D-sampling of render function
  double stepsize; // step size for integration of streamlines
  int stepout; // increment for output of streamline nodes
  int maxsteps; // maximal number of steps
  int numsteps; // current number of steps
  double minmag; // minimal magnitude for stop of integration
  int direction; // direction of streamline integration
  int method; // numerical method of streamline integration
  double thresholddot; // threshold value for dot product of new and old direction for stop of integration

  HeuristicStreamLinesData()
  {
    seed_fieldH=NULL;
    compare_fieldH=NULL;
    src_fieldH=NULL;
    weighting_fieldH=NULL;
    pts_fieldH=NULL;

    sfi=NULL;
    wfi=NULL;

    minper=0.;
    maxper=1.;
    maxlen=0;
    numsamples=1;

    direction=1;
    method=0;
    thresholddot=-.99;
  }

  //find the greatest dimension of the source field
  void boundingBoxMaxLength(FieldHandle fieldH) {
    const BBox bbox = fieldH->mesh()->get_bounding_box();
    if (bbox.valid()) {
      Vector size = bbox.diagonal();
      maxlen = max(size.x(), size.y());
      maxlen = max(maxlen, size.z());
    }
    else
      maxlen=0;
  }
};


class HeuristicStreamLinesAlgo : public DynamicAlgoBase
{
public:
  HeuristicStreamLinesAlgo()
  {
  };

  virtual void execute(HeuristicStreamLinesData &, ProgressReporter * mod )  =0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
                                            const TypeDescription *fvec,
                                            const TypeDescription *fsp
                                            );
};


template <class FSRC, class FVEC, class FSP>
class HeuristicStreamLinesAlgoT
  : virtual public HeuristicStreamLinesAlgo,
    virtual public HeuristicStreamLinesData,
    public SampleFieldRandomAlgoT<typename FSP::mesh_type>
{
public:
  FSRC *src_field;
  typename FSRC::mesh_type *src_mesh;
  FSRC *weighting_field;
  FSP *seed_field;
  FSRC *compare_field;
  typename FSRC::mesh_type *compare_mesh;

  double comparemean;
  double srcmean;

  HeuristicStreamLinesAlgoT()
  {
  }

  //! Set data in a field to zero
  void zeroField(FSRC *field)
  {
    ASSERT(field);

    typename FSRC::fdata_type::iterator itr, end_itr;

    itr=field->fdata().begin();
    end_itr=field->fdata().end();

    while (itr != end_itr)
    {
      *itr=0;
      ++itr;
    }
  }

  virtual void execute(HeuristicStreamLinesData &MESL, ProgressReporter * mod)
  {
    *(HeuristicStreamLinesData *)this=MESL;

    src_field = dynamic_cast<FSRC *>(src_fieldH.get_rep());
    ASSERT(src_field);

    src_mesh = src_field->get_typed_mesh().get_rep();
    ASSERT(src_mesh);

    zeroField(src_field);

    compare_field = dynamic_cast<FSRC *>(compare_fieldH.get_rep());
    ASSERT(compare_field);
    compare_mesh = compare_field->get_typed_mesh().get_rep();

    ASSERT(compare_mesh);
    comparemean=getMean(compare_field);

    boundingBoxMaxLength(src_fieldH);

    weighting_field = dynamic_cast<FSRC *>(weighting_fieldH.get_rep());
    ASSERT(weighting_field);

    seed_field = dynamic_cast<FSP *>(seed_fieldH.get_rep());
    ASSERT(seed_field);

    msl=scinew HSLStreamLines_FSL();
    HSLStreamLines_FSL::mesh_type *mslm=dynamic_cast<typename HSLStreamLines_FSL::mesh_type *>(msl->mesh().get_rep());
    ASSERT(mslm);

    SampleFieldRandomAlgoT<typename FSP::mesh_type> SFalgo;
    unsigned int inumpts = (int)numpts;
    double rmsmin;

    //for each streamline, try placing it in a number of random points, and keep the best one
    for(int i=0; i<numsl; i++) {
      cerr << "\n" << "streamline: "<< i <<"\n";
      rmsmin=HUGE;
      bool found=false;
      FieldHandle mslminH;

      for(int j=0 ; j<inumpts; ++j) {
        cerr<<"trial: "<<j<<"\n";
        FieldHandle rph=SFalgo.execute(mod, seed_fieldH, 1, (int)(pow(2.,31.)*drand48()), "uniuni", 0);

        HSLPoints_FPTS::mesh_type *rp_mesh = dynamic_cast<typename HSLPoints_FPTS::mesh_type *>(rph->mesh().get_rep());
        ASSERT(rp_mesh);
        typename HSLPoints_LPTS::iterator rp_itr;
        rp_mesh->begin(rp_itr);

        Point seed;
        rp_mesh->get_point(seed, *rp_itr);
        //          cerr << seed << endl;
        pts_fieldH=createStreamLine(seed);

        if (numsteps<2)
          j--;
        else { // evaluate the new image
          double rms=evaluateTrial();
          if (rmsmin>rms)
          {
            //            cerr << "FOUND "<<"rms: "<<rms<<"\n";
            rmsmin=rms;
            found=true;
            mslminH=pts_fieldH;
          }
        }

        pts_fieldH=NULL;
      }

      if (found) {
        //          cerr << "HeuristicStreamLinesAlgoT::execute best assigned " << rmsmin << "\n";
        pts_fieldH=mslminH;
        renderPointSet(1);

        typename HSLStreamLines_FSL::mesh_type *mslminm=dynamic_cast<typename HSLStreamLines_FSL::mesh_type *>(mslminH->mesh().get_rep());
        ASSERT(mslminm);
        typename HSLStreamLines_LSL::iterator n_itr, n_end;
        mslminm->begin(n_itr); mslminm->end(n_end);

        typename HSLStreamLines_Mesh::Edge::iterator e_itr, e_end;
        mslminm->begin(e_itr);mslminm->end(e_end);
        if (n_itr!=n_end) {
          Point p;
          mslminm->get_center(p, *n_itr);
          HSLStreamLines_FSL::mesh_type::Node::index_type n1=mslm->add_node(p), n2;
          ++n_itr;
          while (n_itr!=n_end) {
            mslminm->get_center(p, *n_itr);
            n2=mslm->add_node(p);
            if (e_itr!=e_end) {
              HSLStreamLines_FSL::mesh_type::Node::array_type n;
              mslminm->get_nodes(n, *e_itr);
              if (n[0]==*n_itr-1 && n[1]==*n_itr) {
                ++e_itr;
                mslm->add_edge(n1, n2);
              }
            }
            n1=n2;
            ++n_itr;
          }
        }
      }
      mod->update_progress(i+1, numsl);
    }
    msl->resize_fdata();
    cerr << numpts << " " << numsl << " " << rmsmin << "\n";

    MESL=*(HeuristicStreamLinesData *)this;
  }

  double evaluateTrial()
  {
    renderPointSet(1);
    double rms=evaluateRMSDif();
    renderPointSet(-1);
    return rms;
  }

  double evaluateRMSDif()
  {
    typename FSRC::mesh_type::Node::iterator src_itr, src_end_itr;
    src_mesh->begin(src_itr);
    src_mesh->end(src_end_itr);
    typename FSRC::mesh_type::Node::iterator compare_itr, compare_end_itr;
    compare_mesh->begin(compare_itr);
    compare_mesh->end(compare_end_itr);

    srcmean=getMean(src_field);

    double rms=0., srcscale=srcmean, comparescale=comparemean;
    //    cerr << "srcmean: " << srcscale << "\t" << "comparemean: " << comparescale;
    if (srcscale==0) srcscale=1.;
    if (comparescale==0) comparescale=1.;
    while(src_itr != src_end_itr) {
      typename FSRC::value_type srcnorm, comparenorm;
      srcnorm=src_field->fdata()[*src_itr];
      srcnorm=srcnorm/srcscale;
      comparenorm=compare_field->fdata()[*compare_itr];
      comparenorm=comparenorm/comparescale;
      double tmp=srcnorm-comparenorm;
      rms+=tmp*tmp;

      ++src_itr;
      ++compare_itr;
    }
    //  cerr << "\t" << "rms: " << sqrt(rms) << endl;
    return sqrt(rms);
  }

  double getMean(FSRC *field)
  {
    ASSERT(field);

    typename FSRC::mesh_type::Node::iterator itr, end_itr;
    typename FSRC::mesh_type *mesh = field->get_typed_mesh().get_rep();
    ASSERT(mesh);

    mesh->begin(itr);
    mesh->end(end_itr);

    double mean=0, dummy;
    int cnt=0;
    while(itr != end_itr) {
      cnt++;
      bool rc=field->value(dummy, *itr);
      ASSERT(rc);
      //cerr << "dummy: " << dummy<< "\n";
      mean+=dummy;
      ++itr;
    }
    return mean/(cnt ? cnt : 0);
  }

  void renderPointSet(double factor)
  {
    HSLStreamLines_FSL::mesh_type *pts_mesh = dynamic_cast<typename HSLStreamLines_FSL::mesh_type *>(pts_fieldH->mesh().get_rep());
    ASSERT(pts_mesh);

    typename HSLPoints_LPTS::iterator pts_itr, pts_end_itr;
    pts_mesh->begin(pts_itr);
    pts_mesh->end(pts_end_itr);
    //int points=0;

    for( ; pts_itr != pts_end_itr; ++pts_itr) {
      //points++;
      Point p;
      pts_mesh->get_center(p, *pts_itr);
      renderPoint(p, factor);
    }
    //cerr<<"points rendered: " << points << "\n";
  }

  void renderPoint(const Point &seed, double factor)
  {
    //  cerr<<"renderPoint: " << seed << "\n";

    const int& dofs=FSRC::basis_type::dofs();
    Point p;
    double * weights = new double[dofs];
    typename FSRC::mesh_type::Node::array_type na;

    double l;
    wfi->interpolate(l, seed);
    if (l<ming) l=ming;
    else if (l>maxg) l=maxg;

    double maxp=(maxper/100.)*maxlen;
    double minp=(minper/100.)*maxlen;
    double lnorm=(ming==maxg ? 0.5 : (l-ming)/(maxg-ming));
    double radius=(numsamples==1 ? 0. : maxp-(maxp-minp)*lnorm);
    //  if (radius) radius=maxp; // for constant radius tests
    double dradius3=(radius ? 1./(3.*radius*radius) : 1.);

    double ffactor=factor*(radius ? (sqrt(2.*M_PI)/radius) : 1.);
    double inc=(numsamples==1 ? 0. : 2.*radius/(numsamples-1));

    /*       if (factor==1) { */
    /*  cerr << seed << "  "; */
    /*  cerr << "r:" << radius << "\t" << "l:" << l << "\t" << "lnorm:" << lnorm << "\t" << "ffactor:" << ffactor  << endl; */
    /*       } */

    double pz=seed.z();
    double py=seed.y();
    double px=seed.x();

    double az=pz-radius;

    double ww=0;

    for(int i=0; i<numsamples; i++, az+=inc) {
      p.z(az);
      double rz2=pz-az;
      rz2*=rz2;
      double ay=py-radius;

      for(int i=0; i<numsamples; i++, ay+=inc) {
        p.y(ay);
        double ry2=py-ay;
        ry2*=ry2;
        double ax=px-radius;

        for(int i=0; i<numsamples; i++, ax+=inc) {
          p.x(ax);
          double rx2=px-ax;
          rx2*=rx2;
          double val=ffactor*exp(-(rx2+ry2+rz2)*dradius3);

          /*        if (factor==1) { */
          /*          ww+=val; */
          //                  cerr << rx2 << "\t" << ry2 << "\t" << rz2 << "\t" << exp(-(rx2+ry2+rz2)*dradius3) << endl;
          //                  cerr << ax << "\t" << ay << "\t" << az << "\t" << val << endl;
          /*        } */

          int dofs=src_mesh->get_weights(p, na, weights);
          for (unsigned int i = 0; i < dofs; i++) {
            src_field->fdata()[na[i]]+=val*weights[i];

            /*                  double temp=src_field->value(na[i]);   */
            /*                  cerr<< "src data: " << temp << "\n";  */
          }
        }
      }
    }
    delete [] weights;
    /*      if (factor==1)  */
    /*  cerr << ww << endl; */
    //  cerr<<"renderPoint done \n";
  }

  void FindNodes(Point x, double stepsize)
  {
    switch(method) {
    case 0:
      FindEuler(x, stepsize);
      break;

    case 1:
      FindRK2(x, stepsize);
      break;

    case 2:
      FindRK4(x, stepsize);
      break;

    default:
      ASSERT(0);
      break;
    }
  }

  //! interpolate using the generic linear interpolator
  bool interpolate(const Point &p, Vector &v)
  {
    if (vfi->interpolate(v, p)) {
      double a=v.safe_normalize();
      /*         if (a <= minmag) */
      /*           cerr << "Stopped minmag\n"; */
      return a > minmag;
    }

    //cerr << "Stopped interpolation " << p << "\n";
    return false;
  }

  //! add point to streamline
  void addPoint(const int i, const Point &p)
  {
    if (!(i%stepout)) {
      numsteps++;
      const HSLStreamLines_LSL::index_type n1 = cf->get_typed_mesh()->add_node(p);
      if (i && n1)
        cf->get_typed_mesh()->add_edge(n1-1, n1);
    }
  }

  void FindEuler(Point x, double stepsize)
  {
    Vector v0, v0old;
    int i;
    for (i=0; i < maxsteps; i++)
    {
      if (!interpolate(x, v0)) break;
      addPoint(i, x);

      if (i) if (Dot(v0, v0old)<thresholddot)   {
        /*             cerr << "Stopped Dot\n"; */
        break;
      }
      v0old=v0;
      x += stepsize*v0 ;
    }
    /*      cerr << "Stopped after " << i << "\n"; */
  }


  void FindRK2(Point x, double stepsize)
  {
    Vector v0, v0old;
    int i;

    for (i=0; i < maxsteps; i ++) {
      if (!interpolate(x, v0)) break;
      addPoint(i, x);

      if (!interpolate(x + v0*stepsize*0.5, v0)) break;

      if (i) if (Dot(v0, v0old)<thresholddot) {
        /*        cerr << "Stopped Dot\n"; */
        break;
      }
      v0old=v0;
      x += stepsize*v0;
    }
    /*       cerr << "Stopped after " << i << "\n"; */
  }


  void FindRK4(Point x, double stepsize)
  {
    Vector f[4], v0, v0old;
    int i;

    for (i = 0; i < maxsteps; i++) {
      if (!interpolate(x, f[0])) break;
      addPoint(i, x);

      if (!interpolate(x + f[0] * stepsize*0.5, f[1])) break;
      if (!interpolate(x + f[1] * stepsize*0.5, f[2])) break;
      if (!interpolate(x + f[2] * stepsize, f[3])) break;

      v0= f[0] + 2.0 *(f[1] + f[2]) + f[3];
      v0.safe_normalize();

      if (i) if (Dot(v0, v0old)<thresholddot) {
        /*           cerr << "Stopped Dot\n"; */
        break;
      }
      v0old=v0;
      x += stepsize*v0;
    }
    /*       cerr << "Stopped after " << i << "\n"; */
  }

  FieldHandle createStreamLine(Point seed)
  {
    numsteps=0;
    HSLStreamLines_SLHandle cmesh = scinew HSLStreamLines_Mesh();
    cf = new HSLStreamLines_FSL(cmesh);

    // Find the negative streamlines.
    if( direction <= 1 )
      FindNodes(seed, -stepsize);

    // Append the positive streamlines.
    if( direction >= 1 )
      FindNodes(seed, stepsize);

    return cf;
  }

};


class HeuristicStreamLinesGUI : virtual public HeuristicStreamLinesData
{
public:
  GuiInt numsl_;
  GuiInt numpts_;
  GuiDouble minper_;
  GuiDouble maxper_;
  GuiDouble ming_;
  GuiDouble maxg_;
  GuiInt numsamples_;
  GuiInt method_;
  GuiDouble stepsize_;
  GuiInt stepout_;
  GuiInt maxsteps_;
  GuiDouble minmag_;
  GuiInt direction_;

  HeuristicStreamLinesGUI(GuiContext* ctx):
    numsl_(ctx->subVar("numsl")),
    numpts_(ctx->subVar("numpts")),
    minper_(ctx->subVar("minper")),
    maxper_(ctx->subVar("maxper")),
    ming_(ctx->subVar("ming")),
    maxg_(ctx->subVar("maxg")),
    numsamples_(ctx->subVar("numsamples")),
    method_(ctx->subVar("method")),
    stepsize_(ctx->subVar("stepsize")),
    stepout_(ctx->subVar("stepout")),
    maxsteps_(ctx->subVar("maxsteps")),
    minmag_(ctx->subVar("minmag")),
    direction_(ctx->subVar("direction"))
  {
  };

  void LoadGuiVariables()
  {
    numsl=numsl_.get();
    numpts=numpts_.get();
    minper=minper_.get();
    maxper=maxper_.get();
    ming=ming_.get();
    maxg=maxg_.get();
    numsamples=numsamples_.get();
    stepsize = stepsize_.get();
    stepout = stepout_.get();
    maxsteps = maxsteps_.get();
    minmag = minmag_.get();
    direction = direction_.get();
    method = method_.get();
  }
};


class HeuristicStreamLines
  : virtual public HeuristicStreamLinesGUI,
    public Module
{
public:
  HeuristicStreamLines(GuiContext* ctx) :
    HeuristicStreamLinesGUI(ctx),
    Module("HeuristicStreamLines", ctx, Source, "Visualization", "SCIRun")
  {
  }

  virtual ~HeuristicStreamLines()
  {
  }

  virtual void execute()
  {
    //cerr << "HeuristicStreamLines::execute started\n";

    // must find vector field input port
    FieldIPort *vfport = (FieldIPort*)get_iport("Flow");
    if (!vfport) {
      error("Unable to initialize iport 'Flow'.");
      return;
    }

    // the vector field input is required
    FieldHandle vfhandle;
    Field *vf;  // vector field
    if (!vfport->get(vfhandle) || !(vf = vfhandle.get_rep())) {
      return;
    }

    // Check that the flow field input is a vector field.
    vfi = vf->query_vector_interface(this);
    if (!vfi.get_rep()) {
      error("Flow is not a Vector field.");
      return;
    }

    FieldIPort *src_port = (FieldIPort *)get_iport("Source");
    if(!src_port) {
      error("Unable to initialize iport 'Source'.");
      return;
    }
    if (!(src_port->get(src_fieldH) && src_fieldH.get_rep()))
    {
      return;
    }

    // Check that the source field input is a scalar field.
    sfi =src_fieldH.get_rep()->query_scalar_interface(this);
    if (!sfi.get_rep()) {
      error("Source is not a Scalar field.");
      return;
    }

    src_fieldH.detach();
    sfi=src_fieldH.get_rep()->query_scalar_interface(this);

    FieldIPort *weighting_port = (FieldIPort *)get_iport("Weighting");
    if(!weighting_port) {
      error("Unable to get weighting field.");
      return;
    }
    if (!(weighting_port->get(weighting_fieldH) && weighting_fieldH.get_rep()))
    {
      error("Unable to initialize iport 'Weighting'.");
      return;
    }

    wfi =weighting_fieldH.get_rep()->query_scalar_interface(this);
    if (!wfi.get_rep()) {
      error("Weighting is not a scalar field.");
      return;
    }

    FieldIPort *compare_port = (FieldIPort *)get_iport("Compare");
    if(!compare_port) {
      error("Unable to initialize iport 'Compare'.");
      return;
    }
    if (!(compare_port->get(compare_fieldH) && compare_fieldH.get_rep())) {
      error("Unable to get compare field.");
      return;
    }

    cfi =compare_fieldH.get_rep()->query_scalar_interface(this);
    if (!cfi.get_rep()) {
      error("Compare is not a Scalar field.");
      return;
    }

    FieldIPort *seed_port = (FieldIPort *)get_iport("Seed points");
    if(!seed_port) {
      error("Unable to initialize iport 'Seed points'.");
      return;
    }

    if (!(seed_port->get(seed_fieldH) && seed_fieldH.get_rep()))
      seed_fieldH=src_fieldH;

    FieldOPort *sl_oport = (FieldOPort *)get_oport("Streamlines");
    if(!sl_oport) {
      error("Unable to initialize oport 'Streamlines'.");
      return;
    }

    FieldOPort *r_oport = (FieldOPort *)get_oport("Render");
    if(!r_oport) {
      error("Unable to initialize oport 'Render'.");
      return;
    }

    CompileInfoHandle ci =
      HeuristicStreamLinesAlgo::get_compile_info(src_fieldH->get_type_description(),
                                                 vfhandle->get_type_description(),
                                                 seed_fieldH->get_type_description()
                                                 );
    Handle<HeuristicStreamLinesAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    HeuristicStreamLinesGUI::LoadGuiVariables();

    algo->execute(*(HeuristicStreamLinesData *)this, (ProgressReporter *)this);

    sl_oport->send(msl);
    r_oport->send(src_fieldH);
  }
};

} // end namespace SCIRun

#endif // HeuristicStreamLines_h
