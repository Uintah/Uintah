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
 * FILE: NrrdString.cc
 * AUTH: Jeroen Stinstra
 * DATE: 29 Nov  2004
 */
 
 
// This is a simple implementation to use NrrdData objects as strings
// The object simply maintains a pointer to a NrrdData object. 
 
#include <Core/Datatypes/NrrdString.h>

namespace SCIRun {

NrrdString::NrrdString()
{
    nrrdstring_ = 0;
}

NrrdString::NrrdString(const std::string str)
{
    setstring(str);
}

NrrdString::NrrdString(const char *str)
{
    setstring(std::string(str));
}

NrrdString::NrrdString(const NrrdString& nrrdstring)
{
    nrrdstring_ = nrrdstring.nrrdstring_;
}

NrrdString::NrrdString(const NrrdDataHandle& handle)
{
    nrrdstring_ = handle;
}

bool NrrdString::setstring(std::string str)
{
    nrrdstring_ = scinew NrrdData();
    
    if (nrrdstring_.get_rep() == 0) return(false);
    
    if (str.size() > 0)
    {
        // Need to add some error checking here
        nrrdAlloc(nrrdstring_->nrrd, nrrdTypeChar, 1, static_cast<int>(str.size()));
        nrrdAxisInfoSet(nrrdstring_->nrrd, nrrdAxisInfoLabel, "nrrdstring");
        nrrdstring_->nrrd->axis[0].kind = nrrdKindDomain;
        char *val = (char*)nrrdstring_->nrrd->data;
        if (val == 0)
        {
            nrrdstring_ = 0;
            return(false);
        }
        else
        {
            for (size_t p=0;p<str.size();p++) val[p] =str[p];
            return(true);
        }
    }
    return (false);
}

std::string NrrdString::getstring()
{
    std::string str;

    if (nrrdstring_.get_rep() != 0)
    {
        if (nrrdstring_->nrrd != 0)
        {
            if ((nrrdstring_->nrrd->type == nrrdTypeChar)||(nrrdstring_->nrrd->type == nrrdTypeUChar))
            {
                if ((nrrdstring_->nrrd->dim > 0)&&(nrrdstring_->nrrd->axis[0].size > 0))
                {
                    if (nrrdstring_->nrrd->data != 0)
                    {
                        char *ptr = static_cast<char *>(nrrdstring_->nrrd->data);
                        size_t strsize = static_cast<size_t>(nrrdstring_->nrrd->axis[0].size);
                        str.resize(strsize);
                        for (size_t p = 0; p < strsize; p++) str[p] = ptr[p];
                    }
                }
            }
  
            if ((nrrdstring_->nrrd->type == nrrdTypeShort)||(nrrdstring_->nrrd->type == nrrdTypeUShort))
            {
                if ((nrrdstring_->nrrd->dim > 0)&&(nrrdstring_->nrrd->axis[0].size > 0))
                {
                    if (nrrdstring_->nrrd->data != 0)
                    {
                        short *ptr = static_cast<short *>(nrrdstring_->nrrd->data);
                        size_t strsize = static_cast<size_t>(nrrdstring_->nrrd->axis[0].size);
                        str.resize(strsize);
                        for (size_t p = 0; p < strsize; p++) str[p] = static_cast<char>(ptr[p]);
                    }
                }
            }
  
            if ((nrrdstring_->nrrd->type == nrrdTypeInt)||(nrrdstring_->nrrd->type == nrrdTypeUInt))
            {
                if ((nrrdstring_->nrrd->dim > 0)&&(nrrdstring_->nrrd->axis[0].size > 0))
                {
                    if (nrrdstring_->nrrd->data != 0)
                    {
                        int *ptr = static_cast<int *>(nrrdstring_->nrrd->data);
                        size_t strsize = static_cast<size_t>(nrrdstring_->nrrd->axis[0].size);
                        str.resize(strsize);
                        for (size_t p = 0; p < strsize; p++) str[p] = static_cast<char>(ptr[p]);
                    }
                }
            }

            if (nrrdstring_->nrrd->type == nrrdTypeFloat)
            {
                if ((nrrdstring_->nrrd->dim > 0)&&(nrrdstring_->nrrd->axis[0].size > 0))
                {
                    if (nrrdstring_->nrrd->data != 0)
                    {
                        float *ptr = static_cast<float *>(nrrdstring_->nrrd->data);
                        size_t strsize = static_cast<size_t>(nrrdstring_->nrrd->axis[0].size);
                        str.resize(strsize);
                        for (size_t p = 0; p < strsize; p++) str[p] = static_cast<char>(ptr[p]);
                    }
                }
            }

            if (nrrdstring_->nrrd->type == nrrdTypeDouble)
            {
                if ((nrrdstring_->nrrd->dim > 0)&&(nrrdstring_->nrrd->axis[0].size > 0))
                {
                    if (nrrdstring_->nrrd->data != 0)
                    {
                        double *ptr = static_cast<double *>(nrrdstring_->nrrd->data);
                        size_t strsize = static_cast<size_t>(nrrdstring_->nrrd->axis[0].size);
                        str.resize(strsize);
                        for (size_t p = 0; p < strsize; p++) str[p] = static_cast<char>(ptr[p]);
                    }
                }
            }

  
        }
    }
    return(str);
}

NrrdString::~NrrdString()
{
}

} // end namespace
