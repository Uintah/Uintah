/****************************************************************************
** ui.h extension file, included from the uic-generated form implementation.
**
** If you wish to add, delete or rename slots use Qt Designer which will
** update this file, preserving your code. Create an init() slot in place of
** a constructor, and a destroy() slot in place of a destructor.
*****************************************************************************/

#include <iostream.h>
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
    cerr<<"index="<<index<<"s="<<s<<endl;	
    numListBox->insertItem(s, index);
    numLineEdit->clear();
}

void ListForm::refresh()
{

}

void ListForm::del()
{
    int index=numListBox->currentItem();
    numListBox->removeItem(index);
}
