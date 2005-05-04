#include <GL/gl.h>
#include <GL/glut.h>

void display() {
  glClear( GL_COLOR_BUFFER_BIT );

  glColor3f( 1, 1, 0 );
  glBegin( GL_QUADS );
  
  //glTexCoord2f( 0.0, 0.0 );
  //glNormal3f( 0, 0, 1 );
  glVertex2f( -1, -1 );

  //glTexCoord2f( 0.0, 1.0 );
  //glNormal3f( 0, 0, 1 );
  glVertex2f( -1, 1 );

  //glTexCoord2f( 1.0, 0.0 );
  //glNormal3f( 0, 0, 1 );
  glVertex2f( 1, 1 );

  //glTexCoord2f( 1.0, 1.0 );
  //glNormal3f( 0, 0, 1 );
  glVertex2f( 1, -1 );

  glEnd();

  glFlush();
}

void reshape( int w, int h ) {
  glViewport( 0, 0, w, h );
  
  glMatrixMode( GL_PROJECTION );
  glLoadIdentity();
  //gluOrtho2D( 0.0, w, 0.0, h );
  //gluPerspective( 60, w/(float)h,  0.1, 1000 );
  glOrtho( -1, 1, -1, 1, 0, 1000 );
  glMatrixMode( GL_MODELVIEW );
  glLoadIdentity();
  //  glTranslatef( 0, 0, -5 );
}

main( int argc, char ** argv ) {
  
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB | GLUT_DEPTH);
  glutInitWindowSize( 512, 512 );
  glutInitWindowPosition(100,100);
  glutCreateWindow(argv[0]);
  glutDisplayFunc(display);
  glutReshapeFunc(reshape);
  glutMainLoop();
    
}
