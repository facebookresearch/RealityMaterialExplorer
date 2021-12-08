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

public class ECWarningScript : MonoBehaviour
{	
	private float difval;
	private float specval;
    // Start is called before the first frame update
    void Start()
    {
        difval = 0.1f;
		specval = 0.88f;
		gameObject.SetActive(false);
    }
	
	public void updateDifVal(float d){
		difval = d;
		updateECWarning();
	}
	
	public void updateSpecVal(float s){
		specval = s;
		updateECWarning();
	}
	
	private void updateECWarning(){
		bool shouldWarn = (difval+specval>1.0f);
		gameObject.SetActive(shouldWarn);
	}
    // Update is called once per frame
    void Update()
    {
        
    }
}
