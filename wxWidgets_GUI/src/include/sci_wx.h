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

#ifndef include_sci_wx_h
#define include_sci_wx_h

#include <sci_defs/wx_defs.h>

#if defined (HAVE_WX)

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
# include <wx/wx.h>
#endif

// some useful headers
#include <wx/chkconf.h>
#include <wx/dialog.h>
#include <wx/msgdlg.h>
#include <wx/filedlg.h>
#include <wx/dirdlg.h>
#include <wx/textdlg.h>
#include <wx/utils.h>

// WX module checks
#if ! defined(wxUSE_THREADS) || ! defined(wxUSE_STD_IOSTREAM) || ! defined(wxUSE_STD_STRING) || ! defined(wxUSE_LIBPNG) || ! defined(wxUSE_LIBJPEG) || ! defined(wxUSE_MENUS)
#  error("wxWidgets not configured correctly.  Please see build documentation for details.")
#endif // WX module checks

#endif

#endif
