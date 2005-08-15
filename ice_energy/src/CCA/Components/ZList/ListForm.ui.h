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

/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename slots use Qt Designer which will
** update this file, preserving your code. Create an init() slot in place of
** a constructor, and a destroy() slot in place of a destructor.
*****************************************************************************/

#include <iostream>
#include "ZList.h"

void ListForm::enableDelete(int i)
{
    deletePushButton->setEnabled(i>=0);
}

void ListForm::enableInsert(const QString &s)
{
    if(s.isNull()) insertPushButton->setEnabled(false);
    else{		
	    bool ok;
  	  s.toDouble(&ok);
  	  if(!ok) numLineEdit->clear();
    	 insertPushButton->setEnabled(ok);
    } 	
}

void ListForm::insert()
{
    QString s=numLineEdit->text();
    int index=numListBox->currentItem();
    std::cerr<<"index="<<index<<"s="<<s<<std::endl;	
    numListBox->insertItem(s, index);
    numLineEdit->clear();
}

void ListForm::refresh()
{

   std::vector<double> v;
   double size=numListBox->count();
   for(int i=0;i<size;i++){
        v.push_back(numListBox->text(i).toDouble());
   }
   com->datalist=v;
   std::cerr<<"datalist is refreshed with size="<<size<<std::endl;
}

void ListForm::del()
{
    int index=numListBox->currentItem();
    numListBox->removeItem(index);
}

