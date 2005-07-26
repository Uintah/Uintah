/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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

/*
 *  UnuRmapN.cc 
 *
 *  Written by:
 *   Michael Callahan
 *   May 2005
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuRmapN : public Module {
public:
  UnuRmapN(GuiContext*);
  virtual ~UnuRmapN();

  virtual void execute();
};


DECLARE_MAKER(UnuRmapN)
UnuRmapN::UnuRmapN(GuiContext* ctx)
  : Module("UnuRmapN", ctx, Source, "UnuNtoZ", "Teem")
{
}


UnuRmapN::~UnuRmapN()
{
}


void
UnuRmapN::execute()
{
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle dmap_handle;

  update_state(NeedData);
  NrrdIPort *inrrd = (NrrdIPort *)get_iport("InputNrrd");
  NrrdIPort *idmap = (NrrdIPort *)get_iport("RegularMapNrrd");

  if (!inrrd->get(nrrd_handle))
    return;
  if (!idmap->get(dmap_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }
  if (!dmap_handle.get_rep()) {
    error("Empty RegularMapNrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nmap = dmap_handle->nrrd;

  int zerosize = nin->axis[0].size;
  int ninstart = 1;
  if (zerosize > 8) { zerosize = 1; ninstart = 0; }
  
  cout << "zerosize = " << zerosize << "\n";
  cout << "ninstart = " << ninstart << "\n";

  const int offset = nmap->dim - zerosize;
  int noutdim = nin->dim - ninstart + nmap->dim - zerosize;
  int nsize[NRRD_DIM_MAX];
  for (int i = 0; i < offset; i++)
  {
    nsize[i] = nmap->axis[i].size;
    cout << "size1 " << i << " = " << nsize[i] << "\n";
  }
  for (int i = ninstart; i < nin->dim; i++)
  {
    nsize[offset + i - ninstart] = nin->axis[i].size;
    cout << "size2 " << offset + i - ninstart << " = "
         << nsize[offset + i - ninstart] << "\n";
  }
  Nrrd *nout = nrrdNew();
  nrrdAlloc_nva(nout, nmap->type, noutdim, nsize);

  // Copy from nmap->axis[i] to nout->axis[i]
  // Compute colorsize while we're here.
  size_t colorsize = nrrdTypeSize[nmap->type];
  for (int i = 0; i < offset; i++)
  {
    colorsize *= nmap->axis[i].size;

    nout->axis[i].kind = nmap->axis[i].kind;
    nout->axis[i].center = nmap->axis[i].center;
    nout->axis[i].spacing = nmap->axis[i].spacing;
    nout->axis[i].min = nmap->axis[i].min;
    nout->axis[i].max = nmap->axis[i].max;
  }
  // Copy from nin->axis[i] to nout->axix[offset + i - 1].
  for (int i = ninstart; i < nin->dim; i++)
  {
    nout->axis[offset + i - ninstart].kind = nin->axis[i].kind;
    nout->axis[offset + i - ninstart].center = nin->axis[i].center;
    nout->axis[offset + i - ninstart].spacing = nin->axis[i].spacing;
    nout->axis[offset + i - ninstart].min = nin->axis[i].min;
    nout->axis[offset + i - ninstart].max = nin->axis[i].max;
  }


  size_t pixelcount = 1;
  for (int i = ninstart; i < nin->dim; i++)
  {
    pixelcount *= nin->axis[i].size;
  }

  unsigned char *nindata = (unsigned char *)nin->data;
  unsigned char *nmapdata = (unsigned char *)nmap->data;
  unsigned char *noutdata = (unsigned char *)nout->data;
  for (size_t i = 0; i < pixelcount; i++)
  {
    unsigned char *coords = nindata + i * zerosize;
    size_t ncoord = 0;
    size_t off = colorsize;
    for (int j = 0; j < zerosize; j++)
    {
      const int k = (int)((coords[j] / 256.0) * nmap->axis[offset + j].size);
      ncoord += k * off;
      off *= nmap->axis[offset + j].size;
    }
    unsigned char *colors = nmapdata + ncoord;
    unsigned char *result = noutdata + i * colorsize;
    memcpy(result, colors, colorsize);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;
  NrrdDataHandle out(nrrd);

  // Copy the properties.
  out->copy_properties(nrrd_handle.get_rep());

  NrrdOPort *onrrd = (NrrdOPort *)get_oport("OutputNrrd");
  onrrd->send(out);
}


} // End namespace Teem


