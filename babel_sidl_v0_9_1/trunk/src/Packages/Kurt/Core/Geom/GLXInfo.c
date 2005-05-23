void setup_glx12(char* appname, Display *dpy, GLXContext& cx) {
   XVisualInfo *vi;
   Colormap cmap;
   XSetWindowAttributes swa;
   Window win;
   XEvent event;
 
   /* Get an appropriate visual */
   vi = glXChooseVisual(dpy, DefaultScreen(dpy), AttributeList);
   if (!vi) error(appname, "no suitable visual");
   /* Create a GLX context */
   cx = glXCreateContext(dpy, vi, 0, GL_FALSE);
 
   /* Create a colormap */
   cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),vi->visual, AllocNone);
 
   /* Create a window */
   swa.colormap = cmap;
   swa.border_pixel = 0;
   swa.event_mask = StructureNotifyMask;
   win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0,
		       winWidth,  winHeight,
		       0, vi->depth, InputOutput, vi->visual,
		       CWBorderPixel|CWColormap|CWEventMask, &swa);
   XMapWindow(dpy, win);
   XIfEvent(dpy, &event, WaitForNotify, (char*)win);
 
   /* Connect the context to the window */
   glXMakeCurrent(dpy, win, cx);
}
 
void setup_glx13(char* appname, Display *dpy, GLXContext& cx) {
   GLXFBConfig *fbc;
   XVisualInfo *vi;
   Colormap cmap;
   XSetWindowAttributes swa;
   Window win;
   GLXWindow gwin;
   XEvent event;
   int nelements, i, j;
 
   /* Find a FBConfig that uses RGBA.  Note that no attribute list is */
   /* needed since GLX_RGBA_BIT is a default attribute.               */
   fbc = glXChooseFBConfig(dpy, DefaultScreen(dpy), 0, &nelements);
   
   int vals[6];
   int res[] = { 4,4,4,4, 16, GL_FALSE};
   int flags[] = {GLX_RED_SIZE, GLX_GREEN_SIZE, GLX_BLUE_SIZE, GLX_ALPHA_SIZE,
		    GLX_DEPTH_SIZE, GLX_DOUBLEBUFFER };

   for(i = 0; i < nelements; i++){
     int flag = 1;
     for(j = 0; j < 5; j++ ){
       glXGetFBConfigAttrib(dpy, fbc[i], flags[j], &vals[j]);
       if(vals[j] < res[j]) {
	 flag = 0;
	 break;
       }
     }
     if( flag ){
//        glXGetFBConfigAttrib(dpy, fbc[i],flags[j], &vals[j]);
//        if(vals[j] == res[j])
	 break;
     }
   }
   if( i == nelements) i = 0;
   
   vi = glXGetVisualFromFBConfig(dpy, fbc[i]);
   if (!vi) error(appname, "no suitable visual");
   /* Create a GLX context using the first FBConfig in the list. */
   cx = glXCreateNewContext(dpy, fbc[i], GLX_RGBA_TYPE, 0, GL_FALSE);
 
   /* Create a colormap */
   cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen),vi->visual, AllocNone);
 
   /* Create a window */
   swa.colormap = cmap;
   swa.border_pixel = 0;
   swa.event_mask = ExposureMask | StructureNotifyMask | KeyPressMask |
     ButtonPressMask | ButtonMotionMask ;
  win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0,
		      winWidth,  winHeight,
		      0, vi->depth, InputOutput, vi->visual,
		      CWBorderPixel|CWColormap|CWEventMask, &swa);

   XMapWindow(dpy, win);
   XIfEvent(dpy, &event, WaitForNotify, (char*)win);
 
   /* Create a GLX window using the same FBConfig that we used for the */
   /* the GLX context.                                                 */
   gwin = glXCreateWindow(dpy, fbc[i], win, 0);
 
   /* Connect the context to the window for read and write */
   glXMakeContextCurrent(dpy, gwin, gwin, cx);
}
