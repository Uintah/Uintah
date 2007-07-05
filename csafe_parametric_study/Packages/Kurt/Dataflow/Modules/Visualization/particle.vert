//
// Vertex shader for rendering a "confetti cannon"
// via a partcle system
//
// Author: Randi Rost
//
// Copyright (c) 2003-2004: 3Dlabs, Inc.
//
// See 3Dlabs-License.txt for license information
//

uniform vec4  Background;      // constant color equal to background 
//uniform vec4  StartPosition;
varying vec4 Color;
 
void main(void)
{
    vec4  vert = gl_Vertex;
    float t = vert.w;

    

    if (t >= 0.0)
    {
        Color   = gl_Color;
    }
    else
    {
        Color = Background;     // "pre-birth" color
    }
 
    gl_Position  = gl_ModelViewProjectionMatrix * vert;
}