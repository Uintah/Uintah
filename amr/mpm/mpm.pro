#-------------------------------------------------
#
# Project created by QtCreator 2014-08-20T17:48:46
#
#-------------------------------------------------
QMAKE_CXXFLAGS += -std=c++0x
QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = mpm
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    InitialConditions.cpp \
    Utils.cpp \
    Solver.cpp \
    Particle.cpp \
    BoundingBox.cpp \
    Basis.cpp \
    Node.cpp \
    MPMView.cpp \
    Mesh.cpp

HEADERS  += mainwindow.h \
    InitialConditions.h \
    Utils.h \
    Solver.h \
    Particle.h \
    BoundingBox.h \
    Basis.h \
    Node.h \
    Typedefs.h \
    MPMView.h \
    Mesh.h

FORMS    += mainwindow.ui
