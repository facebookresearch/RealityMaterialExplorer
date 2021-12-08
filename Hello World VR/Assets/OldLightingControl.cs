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

public class OldLightingControl : MonoBehaviour
{	public GameObject probe_grace;
	public GameObject probe_uffizi;
	public GameObject probe_peters;
	public GameObject probe_grove;
	public Material map_void;
	public Material map_grace;
	public Material map_uffizi;
	public Material map_peters;
	public Material map_grove;
	public GameObject point_light;
    // Start is called before the first frame update
    void Start()
    {
        
    }
	
	public void chooseVoid(){
		RenderSettings.skybox=map_void;
		probe_grace.SetActive(false);
		probe_uffizi.SetActive(false);
		probe_peters.SetActive(false);
		probe_grove.SetActive(false);
		point_light.SetActive(true);
	}
	
	public void chooseGrace(){
		RenderSettings.skybox=map_grace;
		probe_grace.SetActive(true);
		probe_uffizi.SetActive(false);
		probe_peters.SetActive(false);
		probe_grove.SetActive(false);
		point_light.SetActive(false);
	}
	
	public void chooseUffizi(){
		RenderSettings.skybox=map_uffizi;
		probe_grace.SetActive(false);
		probe_uffizi.SetActive(true);
		probe_peters.SetActive(false);
		probe_grove.SetActive(false);
		point_light.SetActive(false);
	}
	
	public void choosePeters(){
		RenderSettings.skybox=map_peters;
		probe_grace.SetActive(false);
		probe_uffizi.SetActive(false);
		probe_peters.SetActive(true);
		probe_grove.SetActive(false);
		point_light.SetActive(false);
	}
	
	public void chooseGrove(){
		RenderSettings.skybox=map_grove;
		probe_grace.SetActive(false);
		probe_uffizi.SetActive(false);
		probe_peters.SetActive(false);
		probe_grove.SetActive(true);
		point_light.SetActive(false);
	}

    // Update is called once per frame
    void Update()
    {
        
    }
}
