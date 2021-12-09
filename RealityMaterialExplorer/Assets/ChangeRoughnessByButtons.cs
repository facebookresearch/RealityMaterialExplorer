/*
Copyright (c) Meta Platforms, Inc. and affiliates.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

using System.Collections;
using System.Collections.Generic;
using UnityEngine;



public class ChangeRoughnessByButtons : MonoBehaviour
{
	
	public Material BRDF1Material;
	public Material BRDF2Material;
	public Material BRDF3Material;
	
	
	float smoothness = 0.9f;
    // Start is called before the first frame update
    void Start()
    {
        
    }
	

	public void SetRoughness(float alpha){
		float smoothness = 1-alpha;
		BRDF1Material.SetFloat("_GlossMapScale",smoothness);
		BRDF2Material.SetFloat("_GlossMapScale",smoothness);
		BRDF3Material.SetFloat("_GlossMapScale",smoothness);
		
	}
	
	public void SetSpecularScale(float s){
		Color speccolor = s*Color.white;//new Color(d,d,d,1.0f);
		BRDF1Material.SetColor("_SpecColor",speccolor);
		BRDF2Material.SetColor("_SpecColor",speccolor);
		BRDF3Material.SetColor("_SpecColor",speccolor);
		
	}
	
	public void SetDiffuseScale(float d){
		Color difcolor = d*Color.white;//new Color(d,d,d,1.0f);
		BRDF1Material.SetColor("_Color",difcolor);
		BRDF2Material.SetColor("_Color",difcolor);
		BRDF3Material.SetColor("_Color",difcolor);
		
	}
	
	public void SetGeometry(int i){
		if(false);
		
		
		
	}
    // Update is called once per frame
    void Update()
    {
		
    }
}
