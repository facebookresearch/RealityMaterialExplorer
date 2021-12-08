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
using UnityEngine.UI;

public class MERLQuickSelect : MonoBehaviour
{
	
	public string slot;
	public int index;
	public MasterChangeActiveObject target;
    // Start is called before the first frame update
    void Start()
    {
		index = PlayerPrefs.GetInt(slot+"_matnum",index);
        GetComponentInChildren<Text>().text = target.getMERLname(index);
    }
	public void quickSave(){
		index = target.getMERLnumber();
		GetComponentInChildren<Text>().text = target.getMERLname(index);
		PlayerPrefs.SetInt(slot+"_matnum",index);
	}
	
	public void quickSelect(){
		target.changeMERLMaterial(index);
	}
    // Update is called once per frame
    void Update()
    {
        
    }
}
