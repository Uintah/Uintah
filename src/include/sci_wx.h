/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#ifndef include_sci_wx_h
#define include_sci_wx_h

#include <sci_defs/wx_defs.h>

#if defined (HAVE_WX)

#define HAVE_GUI 1

#include <wx/wxprec.h>
#ifndef WX_PRECOMP
#include <wx/wx.h>
#endif

// some useful headers
#include <wx/dialog.h>
#include <wx/msgdlg.h>
#include <wx/filedlg.h>
#include <wx/dirdlg.h>
#include <wx/textdlg.h>
#include <wx/numdlg.h>
#include <wx/utils.h>
#include <wx/string.h>
#include <wx/strconv.h>

#include <string>

wxString STLTowxString(const std::string& s);
std::string wxToSTLString(const wxString& wxs);

// Wrapping STL strings with Unicode support is problematic
// at this time because the SIDL compiler does not have std::wstring
// support.
// TODO: Revisit this issue during Babel compiler changeover.

#if wxUSE_STD_STRING && ! wxUSE_UNICODE
inline wxString STLTowxString(const std::string& s)
{
  return wxString(s);
}

inline std::string wxToSTLString(const wxString& wxs)
{
  return (std::string) wxs;
}
#else
# if wxUSE_UNICODE
inline wxString STLTowxString(const std::string& s)
{
  return wxString(s.c_str(), *wxConvCurrent);
}

inline std::string wxToSTLString(const wxString& wxs)
{
  return std::string((const char*) wxs.mb_str(*wxConvCurrent));
}
# else // ANSI
inline wxString STLTowxString(const std::string& s)
{
  return wxString(s.c_str());
}

inline std::string wxToSTLString(const wxString& wxs)
{
  return std::string(wxs.c_str());
}
# endif
#endif // wxUSE_STD_STRING

#endif // HAVE_WX

#endif
