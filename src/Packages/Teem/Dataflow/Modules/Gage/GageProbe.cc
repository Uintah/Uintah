/*
 *  GageProbe.cc:
 *
 *  Written by:
 *   Anastasia Mironova
 *   TODAYS DATE HERE
 *
 * TODO: 
 * - take input from GUI to set gageKind;
 * - figure out how to set gageKind without having to read it from the gui
 *
 */

#include <teem/air.h>
#include <teem/gage.h>
#include <teem/nrrd.h>

#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <sstream>
#include <iostream>

#define SPACING(spc) (AIR_EXISTS(spc) ? spc: nrrdDefaultSpacing)

namespace SCITeem {

using namespace SCIRun;

class GageProbe : public Module {
public:
  GageProbe(SCIRun::GuiContext *ctx);
  virtual ~GageProbe();
  virtual void execute();

private:
  NrrdIPort* inrrd_;
  NrrdOPort* onrrd_;
  
  void setGageKind(gageKind *& kind, gageKind *newkind);
  
  GuiString field_kind_;
  GuiString otype_;
  GuiString quantity_;
  
  GuiString valuesType_;
  GuiString valuesNumParm1_;
  GuiString valuesNumParm2_;
  GuiString valuesNumParm3_;
  
  GuiString dType_;
  GuiString dNumParm1_;
  GuiString dNumParm2_;
  GuiString dNumParm3_;
  
  GuiString ddType_;
  GuiString ddNumParm1_;
  GuiString ddNumParm2_;
  GuiString ddNumParm3_;
}; 
  
DECLARE_MAKER(GageProbe)
GageProbe::GageProbe(SCIRun::GuiContext *ctx)
  : Module("GageProbe", ctx, Source, "Gage", "Teem"),
  field_kind_(get_ctx()->subVar("field_kind_")),
  otype_(get_ctx()->subVar("otype_")),
  quantity_(get_ctx()->subVar("quantity_")),
  
  valuesType_(get_ctx()->subVar("valuesType_")),
  valuesNumParm1_(get_ctx()->subVar("valuesNumParm1_")),
  valuesNumParm2_(get_ctx()->subVar("valuesNumParm2_")),
  valuesNumParm3_(get_ctx()->subVar("valuesNumParm3_")),
  
  dType_(get_ctx()->subVar("dType_")),
  dNumParm1_(get_ctx()->subVar("dNumParm1_")),
  dNumParm2_(get_ctx()->subVar("dNumParm2_")),
  dNumParm3_(get_ctx()->subVar("dNumParm3_")),
  
  ddType_(get_ctx()->subVar("ddType_")),
  ddNumParm1_(get_ctx()->subVar("ddNumParm1_")),
  ddNumParm2_(get_ctx()->subVar("ddNumParm2_")),
  ddNumParm3_(get_ctx()->subVar("ddNumParm3_"))
{
  string result;
  string input = "{Scalar Vector}";
  get_gui()->eval(get_id() + " set_list " + input, result);
  printf("result is %s\n", result.c_str());
}


GageProbe::~GageProbe()
{
}


void
GageProbe::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");
  
  if (!inrrd_->get(nrrd_handle))
    return;
  
  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  
  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();
  
  //Set the GUI variables
  field_kind_.reset();
  otype_.reset();
  quantity_.reset();
  
  valuesType_.reset();
  valuesNumParm1_.reset();
  valuesNumParm2_.reset();
  valuesNumParm3_.reset();
  
  dType_.reset();
  dNumParm1_.reset();
  dNumParm2_.reset();
  dNumParm3_.reset();
  
  ddType_.reset();
  ddNumParm1_.reset();
  ddNumParm2_.reset();
  ddNumParm3_.reset();
  
  
  gageContext *ctx;
  double gmc, ipos[4], /*opos[4], */ minx, miny, minz, spx, spy, spz;
  float x, y, z;
  float scale[3];
  gageKind *kind = NULL;
  int a, ansLen, E=0, idx, otype, /*renorm,*/ what;
  int six, siy, siz, sox, soy, soz, xi, yi, zi;
  int iBaseDim, oBaseDim;
  gagePerVolume *pvl;
  char /* *outS,*/ *err = NULL;
  NrrdKernelSpec *k00 = NULL, *k11 = NULL, *k22 = NULL;
  
  //attempt to set gageKind
  if (nin->axis[0].size == 1){
    //first axis has only one value, guess it's a scalar field
    setGageKind(kind, gageKindScl);
  } else if (nin->axis[0].kind == nrrdKindScalar){
    //kind set explicitly in nrrd object, guess it's a scalar field
    setGageKind(kind, gageKindScl);
  } else if (field_kind_.get() == "Scalar"){
    warning("Field Kind is not set in the input Nrrd, making a guess based "\
    "upon the GUI settings.");
    setGageKind(kind, gageKindScl);
  } else if (nin->axis[0].size == 3){
    //first axis has three values, guess it's a vector field  (%p)\n",kind
    setGageKind(kind, gageKindVec);
  } else if (nin->axis[0].kind == nrrdKind3Vector){
    //printf("kind set explicitly in nrrd object, guess it's a vector field\n");
    setGageKind(kind, gageKindVec);
  } else if (field_kind_.get() == "Vector"){
    warning("Field Kind is not set in the input Nrrd, making a guess based "\
    "upon the GUI settings.");
    setGageKind(kind, gageKindVec);
  } else {
    error("Cannot set gageKind.");
    return;
  }
  
  //set the type of output nrrd
  if (otype_.get() == "double") {
    otype = nrrdTypeDouble;
  } else if (otype_.get() == "float") {
    otype = nrrdTypeFloat;
  } else {
    otype = nrrdTypeDefault;
  }
  
  what = airEnumVal(kind->enm, quantity_.get().c_str());
  if (-1 == what) {
    /* -1 indeed always means "unknown" for any gageKind */
    string err = "couldn't parse " + quantity_.get() + " as measure of ";
    char cerr[] = "";
    strcat(cerr, err.c_str());
    strcat(cerr, kind->name);
    strcat(cerr, " volume.");
    error(cerr);
    return;
  }
  
  //set min grad magnitude, for curvature-based queries, use zero when 
  //gradient is below this
  gmc = 0.0;
  
  //set scaling factor for resampling on each axis
  scale[0] = 1.0;
  scale[1] = 1.0;
  scale[2] = 1.0;
  
  k00 = nrrdKernelSpecNew();
  k11 = nrrdKernelSpecNew();
  k22 = nrrdKernelSpecNew();
  
  //set the nrrd kernels' parameters
  string k00parms = "";
  k00parms += valuesType_.get();
  k00parms += ":";
  k00parms += valuesNumParm1_.get();
  if (valuesNumParm2_.get() != ""){
    k00parms += ",";
    k00parms += valuesNumParm2_.get();
  }
  if (valuesNumParm3_.get() != ""){
    k00parms += ",";
    k00parms += valuesNumParm3_.get();
  }
  
  string k11parms = "";
  k11parms += dType_.get();
  k11parms += ":";
  k11parms += dNumParm1_.get();
  if (dNumParm2_.get() != ""){
    k11parms += ",";
    k11parms += dNumParm2_.get();
  }
  if (dNumParm3_.get() != ""){
    k11parms += ",";
    k11parms += dNumParm3_.get();
  }
  
  string k22parms = "";
  k22parms += ddType_.get();
  k22parms += ":";
  k22parms += ddNumParm1_.get();
  if (ddNumParm2_.get() != ""){
    k22parms += ",";
    k22parms += ddNumParm2_.get();
  }
  if (ddNumParm3_.get() != ""){
    k22parms += ",";
    k22parms += ddNumParm3_.get();
  }
  
  nrrdKernelSpecParse(k00, k00parms.c_str());
  nrrdKernelSpecParse(k11, k11parms.c_str());
  nrrdKernelSpecParse(k22, k22parms.c_str());
  
  ansLen = kind->table[what].answerLength;
  iBaseDim = kind->baseDim;
  oBaseDim = 1 == ansLen ? 0 : 1;
  six = nin->axis[0+iBaseDim].size;
  siy = nin->axis[1+iBaseDim].size;
  siz = nin->axis[2+iBaseDim].size;
  spx = SPACING(nin->axis[0+iBaseDim].spacing);
  spy = SPACING(nin->axis[1+iBaseDim].spacing);
  spz = SPACING(nin->axis[2+iBaseDim].spacing);
  sox = (int)scale[0]*six;
  soy = (int)scale[1]*siy;
  soz = (int)scale[2]*siz;
  nin->axis[0+iBaseDim].spacing = SPACING(nin->axis[0+iBaseDim].spacing);
  nin->axis[1+iBaseDim].spacing = SPACING(nin->axis[1+iBaseDim].spacing);
  nin->axis[2+iBaseDim].spacing = SPACING(nin->axis[2+iBaseDim].spacing);
  
  minx = nin->axis[0+iBaseDim].min;
  miny = nin->axis[1+iBaseDim].min;
  minz = nin->axis[2+iBaseDim].min;
  
  //set up gage
  ctx = gageContextNew();
  gageParmSet(ctx, gageParmGradMagCurvMin, gmc);
  gageParmSet(ctx, gageParmVerbose, 1);
  gageParmSet(ctx, gageParmRenormalize, AIR_FALSE);
  gageParmSet(ctx, gageParmCheckIntegrals, AIR_TRUE);
  E = 0;
  if (!E) E |= !(pvl = gagePerVolumeNew(ctx, nin, kind));
  if (!E) E |= gagePerVolumeAttach(ctx, pvl);
  if (!E) E |= gageKernelSet(ctx, gageKernel00, k00->kernel, k00->parm);
  if (!E) E |= gageKernelSet(ctx, gageKernel11, k11->kernel, k11->parm); 
  if (!E) E |= gageKernelSet(ctx, gageKernel22, k22->kernel, k22->parm);
  if (!E) E |= gageQueryItemOn(ctx, pvl, what);
  if (!E) E |= gageUpdate(ctx);
  if (E) {
    error(biffGet(GAGE));
    return;
  }
  const gage_t *answer = gageAnswerPointer(ctx, pvl, what);
  gageParmSet(ctx, gageParmVerbose, 0);
  //end gage setup
  
  if (ansLen > 1) {
    printf("creating %d x %d x %d x %d output\n", 
	   ansLen, sox, soy, soz);
    size_t size[NRRD_DIM_MAX];
    size[0] = ansLen; size[1] = sox;
    size[2] = soy;    size[3] = soz;
    if (!E) E |= nrrdMaybeAlloc_nva(nout=nrrdNew(), otype, 4, size);
  } else {
    size_t size[NRRD_DIM_MAX];
    size[0] = sox; size[1] = soy; size[2] = soz;
    printf("creating %d x %d x %d output\n", sox, soy, soz);
    if (!E) E |= nrrdMaybeAlloc_nva(nout=nrrdNew(), otype, 3, size);
  }
  if (E) {
    error(err);
    return;
  }
  
  //probing the volume
  for (zi=0; zi<=soz-1; zi++) {
    z = AIR_AFFINE(0, zi, soz-1, 0, siz-1);
    for (yi=0; yi<=soy-1; yi++) {
      y = AIR_AFFINE(0, yi, soy-1, 0, siy-1);
      for (xi=0; xi<=sox-1; xi++) {
	x = AIR_AFFINE(0, xi, sox-1, 0, six-1);
        idx = xi + sox*(yi + soy*zi);
   	
	ipos[0] = xi;
	ipos[1] = yi;
	ipos[2] = zi;
	
	if (gageProbe(ctx, ipos[0], ipos[1], ipos[2])) {
          error(ctx->errStr);
        }
        
	if (1 == ansLen) {	
	    nrrdFInsert[nout->type](nout->data, idx, nrrdFClamp[nout->type](*answer));
        } else {
          for (a=0; a<=ansLen-1; a++) {
            nrrdFInsert[nout->type](nout->data, a + ansLen*idx, 
                                    nrrdFClamp[nout->type](answer[a]));
          }
        }	
      }
    }
  }
  
  nrrdContentSet_va(nout, "probe", nin, "%s", airEnumStr(kind->enm, what));
  nout->axis[0+oBaseDim].spacing = 
    ((double)six/sox)*SPACING(nin->axis[0+iBaseDim].spacing);
  nout->axis[0+oBaseDim].label = airStrdup(nin->axis[0+iBaseDim].label);
  nout->axis[1+oBaseDim].spacing = 
    ((double)six/sox)*SPACING(nin->axis[1+iBaseDim].spacing);
  nout->axis[1+oBaseDim].label = airStrdup(nin->axis[1+iBaseDim].label);
  nout->axis[2+oBaseDim].spacing = 
    ((double)six/sox)*SPACING(nin->axis[2+iBaseDim].spacing);
  nout->axis[2+oBaseDim].label = airStrdup(nin->axis[2+iBaseDim].label);
  
  
  for (unsigned int i = 0; i < nout->dim; i++)
  {
    if (!(airExists(nout->axis[i].min) && 
	  airExists(nout->axis[i].max)))
      nrrdAxisInfoMinMaxSet(nout, i, nrrdCenterNode);
  }
  
  //send the nrrd to the output
  NrrdDataHandle ntmp(scinew NrrdData(nout));
  onrrd_->send_and_dereference(ntmp);
}


void
GageProbe::setGageKind(gageKind *& kind, gageKind *newkind)
{
  kind = newkind;
}

} //End namespace SCITeem

