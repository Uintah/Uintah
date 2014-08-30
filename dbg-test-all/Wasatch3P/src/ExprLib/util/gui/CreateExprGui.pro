#-------------------------------------------------
#
# Project created by QtCreator 2011-07-15T17:33:17
#
#-------------------------------------------------

QT       += core gui

TARGET = CreateExprGui
TEMPLATE = app

INCLUDEPATH += ../ \
               ../../


SOURCES += main.cpp\
           EntryList.cpp \
           mainwindow.cpp \
           Highlighter.cpp \
           vartable.cpp \
           exprnamedialog.cpp \
           ../CreateExpr.cpp \
    ../../expression/Context.cpp \
    ../../expression/Tag.cpp

HEADERS  += EntryList.h \
            mainwindow.h \
            Highlighter.h \
            vartable.h \
            exprnamedialog.h \
            ../CreateExpr.h \
    ../../expression/Context.h \
    ../../expression/Tag.h

FORMS    += EntryList.ui \
            mainwindow.ui \
            vartable.ui \
            exprnamedialog.ui

OTHER_FILES += \
    refresh.tiff

RESOURCES += \
    icons.qrc
