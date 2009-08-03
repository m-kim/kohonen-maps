#version 120
#extension GL_EXT_geometry_shader4 : enable
varying vec3 normal;
varying out vec3 oNormal;
varying 
void drawTop(vec4 position)
{
    vec4 v1, v2, v3, tmp;
    vec3 v_n1, v_n2;

		v1 = position + vec4(-.5,-.5,0,0);
		v2 = position + vec4(-.5,.5,0,0);
		v3 = position + vec4(.5,-.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n2, v_n1));
		gl_Position = v1;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
		gl_Position = v2;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
		v1 = v2;
		v2 = v3;
		v3 = position + vec4(.5,.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		gl_Position = v3; 
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();

		EndPrimitive();
}

void setLight(vec4 v2)
{
	vec3 light_dir;
	float dot_prod;
	vec4 vector_pos = vec4(gl_NormalMatrix * v2.xyz, v2.w);
	
	light_dir = normalize(vec3(vector_pos - gl_LightSource[0].position) );
	dot_prod = max(dot(oNormal, light_dir),0.0);
	//gl_FrontColor =  vec4(light_dir, 1.0);
	gl_FrontColor = gl_FrontColor * dot_prod;

}
void main(void)
{
  vec4 v1, v2, v3, tmp;
  vec3 v_n1, v_n2;
	int i;
	for(i=0; i< gl_VerticesIn; i++){
		//drawTop(gl_PositionIn[i]);

		gl_FrontColor = gl_FrontColorIn[i];
		v1 = gl_PositionIn[i] + vec4(-.5,-.5,0,0);
		v2 = gl_PositionIn[i] + vec4(-.5,-.5,-gl_PositionIn[i].z,0);
		v3 = gl_PositionIn[i] + vec4(.5,-.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(gl_NormalMatrix * cross(v_n2, v_n1));
		setLight(v2);
    gl_Position = v1;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		gl_Position = v2;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();

		
    v1 = v2;
    v2 = v3;
		v3 = gl_PositionIn[i] + vec4(.5,-.5,-gl_PositionIn[i].z,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		setLight(v2);
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		
		
    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(.5,.5,0,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		setLight(v2);
		gl_Position = v3; 
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();
    
    
    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(.5,.5,-gl_PositionIn[i].z,0);
		v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		setLight(v2);
		gl_Position = v3; 
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();


    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,.5,-gl_PositionIn[i].z,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		setLight(v2);
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();


    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,.5,0,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n2, v_n1));
		setLight(v2);
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();

		
		v1 = v2;
		v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,-.5,-gl_PositionIn[i].z,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		setLight(v2);
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
    EmitVertex();

    
    v1 = v2;
    v2 = v3;
    v3 = gl_PositionIn[i] + vec4(-.5,-.5,0,0);
    v_n1 = v3.xyz - v2.xyz;
		v_n2 = v1.xyz - v2.xyz;
		oNormal = normalize(cross(v_n1, v_n2));
		setLight(v2);
		gl_Position = v3;
		gl_Position = gl_ModelViewProjectionMatrix * gl_Position;
		EmitVertex();
		EndPrimitive();
	}
}