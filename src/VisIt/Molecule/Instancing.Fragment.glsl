#version 120
#extension EXT_gpu_shader4 : enable


varying vec3 eye_position;
varying vec3 eye_normal;
varying vec3 data_attribute;
void main()
{
	vec3 N = normalize(eye_normal);
	vec3 V = -normalize(eye_position);

	vec3 L = normalize(gl_LightSource[0].position.xyz - eye_position);

	vec3 H = normalize( V + L);
	float NdotL = max(0,dot(N,L));
	float NdotH = max(0,dot(N,H));

	vec3 surface_color = vec3(data_attribute);
	
	float diffuse = max(0, NdotL);
	float specular = NdotL > 0 ? pow(NdotH, gl_FrontMaterial.shininess) : 0;

	float attenuation = 1.0f;

	vec3 color;
	color = 
		gl_FrontLightModelProduct.sceneColor.xyz +
		gl_LightSource[0].ambient.xyz * attenuation * surface_color +
		gl_LightSource[0].diffuse.xyz * diffuse * surface_color;
	color += gl_LightSource[0].specular.xyz * specular * attenuation *  gl_FrontMaterial.specular.xyz;
	color = clamp( color, 0.0, 1.0 );

	gl_FragColor = vec4(color,1);
}
